"""The global deep arm -- N-BEATS, the optional model the spec names.

WHY N-BEATS AND NOT DeepAR
--------------------------
Both were on the table. N-BEATS is the more honest choice for *this* comparison
because it is a pure point forecaster with no distributional head, so it competes
with the GBM on exactly the same terms. DeepAR outputs a distribution, which would
make it look better on interval metrics for reasons that have nothing to do with
whether its point forecast is any good -- and the interval question is already
answered properly by the quantile GBMs in `quantiles.py`.

WHAT IT IS
----------
Generic N-BEATS: a stack of fully-connected blocks, each producing a *backcast*
(its reconstruction of the input window) and a *forecast*. Each block subtracts
its backcast from the running residual, so block k only ever sees what blocks
1..k-1 failed to explain. That residual stacking is the whole architecture; there
is no convolution, no attention and no recurrence, which is why it trains on CPU.

THE HONEST FRAMING
------------------
This is the arm most likely to disappoint, and it should be reported that way
rather than tuned until it wins. Deep global models earn their keep on tens of
thousands of related series; there are 300 here. A neural net that loses to a GBM
on 300 series is not evidence that neural nets lose -- it is evidence that this
dataset is below the size where they start paying, which is itself the useful
finding for anyone deciding whether to staff the work.

The window is 56 days and the horizon 28, matching the GBM's direct multi-horizon
setup exactly, so the two are scored on the same folds with the same origins.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

WINDOW = 56
HORIZON = 28


class Block(nn.Module):
    """One N-BEATS generic block: shared trunk, two linear heads."""

    def __init__(self, window: int, horizon: int, width: int = 256, depth: int = 3):
        super().__init__()
        layers, d = [], window
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.ReLU()]
            d = width
        self.trunk = nn.Sequential(*layers)
        self.backcast = nn.Linear(width, window)
        self.forecast = nn.Linear(width, horizon)

    def forward(self, x):
        h = self.trunk(x)
        return self.backcast(h), self.forecast(h)


class NBeats(nn.Module):
    def __init__(self, window: int = WINDOW, horizon: int = HORIZON,
                 n_blocks: int = 3, width: int = 256):
        super().__init__()
        self.blocks = nn.ModuleList(
            [Block(window, horizon, width) for _ in range(n_blocks)])

    def forward(self, x):
        residual = x
        total = torch.zeros(x.shape[0], self.blocks[0].forecast.out_features,
                            device=x.device, dtype=x.dtype)
        for b in self.blocks:
            back, fore = b(residual)
            residual = residual - back      # the residual stacking
            total = total + fore
        return total


# --------------------------------------------------------------------------
# scaling
# --------------------------------------------------------------------------
def _scale(win: np.ndarray) -> np.ndarray:
    """Per-window mean scaling, the standard N-BEATS/M4 treatment.

    Without it the loss is dominated by the handful of high-volume series and
    the network never learns the shape of the slow movers -- which on a retail
    assortment is most of the assortment. The scale is restored on output, so
    the model learns SHAPE and the level is handed back to it.
    """
    return np.clip(win.mean(axis=1, keepdims=True), 0.05, None)


def make_windows(panel: np.ndarray, window: int = WINDOW, horizon: int = HORIZON,
                 stride: int = 3):
    """panel: (n_series x T) unit sales. Returns scaled (X, Y) and the scales.

    Only windows whose ENTIRE horizon lies before the forecast origin are built
    by the caller slicing `panel` first -- this function has no notion of the
    origin and will happily use everything it is given, so the caller owns the
    leakage boundary.
    """
    Xs, Ys = [], []
    T = panel.shape[1]
    for s in range(panel.shape[0]):
        y = panel[s]
        for t in range(window, T - horizon + 1, stride):
            Xs.append(y[t - window:t])
            Ys.append(y[t:t + horizon])
    X = np.asarray(Xs, np.float32)
    Y = np.asarray(Ys, np.float32)
    sc = _scale(X).astype(np.float32)
    return X / sc, Y / sc, sc


def train(X, Y, epochs: int = 12, batch: int = 512, lr: float = 1e-3,
          seed: int = 0, device: str = "cpu") -> NBeats:
    torch.manual_seed(seed)
    model = NBeats().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    Xt = torch.tensor(X, device=device)
    Yt = torch.tensor(Y, device=device)
    n = len(Xt)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        model.train()
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            pred = model(Xt[idx])
            # MAE in scaled space == the sample-weighted MASE-shaped loss the
            # M4/M5 literature uses; squared error here would chase the spikes.
            loss = (pred - Yt[idx]).abs().mean()
            loss.backward()
            opt.step()
        sched.step()
    return model


def forecast(model: NBeats, history: np.ndarray, window: int = WINDOW,
             device: str = "cpu") -> np.ndarray:
    """history: (n_series x >=window). Returns (n_series x horizon), unscaled."""
    win = np.asarray(history, np.float32)[:, -window:]
    sc = _scale(win).astype(np.float32)
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(win / sc, device=device)).cpu().numpy()
    return np.clip(out * sc, 0, None)
