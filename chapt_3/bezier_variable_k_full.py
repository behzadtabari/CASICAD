"""
train_and_test_bezier_varK.py

- Loads fixed dataset (.npz created by make_bezier_dataset.py)
- Trains model
- Tests each epoch (proper model.eval() + torch.no_grad())
- Saves model

Run:
  python train_and_test_bezier_varK.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


# ============================
# CONFIG
# ============================

DATA_PATH = "dataset_bezier_varK_D2_N48_M8.npz"

EPOCHS = 40
STEPS_PER_EPOCH = 400
BATCH_SIZE = 128
LR = 1e-3
LAMBDA_SIMPLE = 0.08
NCURVE = 64

SEED = 0


# ============================
# Utils
# ============================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def normalize_points(Q, eps=1e-8):
    """
    Q: (B,N,D) -> normalized Qn: (B,N,D)
    """
    mu = Q.mean(dim=1, keepdim=True)
    Qc = Q - mu
    s = torch.sqrt((Qc**2).sum(-1).mean(1, keepdim=True)).unsqueeze(-1)
    s = torch.clamp(s, min=eps)
    return Qc / s


# ============================
# Geometry
# ============================

def lerp(a, b, t):
    return (1 - t) * a + t * b


def de_casteljau(P, t):
    """
    P: (B,K,D)
    t: scalar OR (B,) OR (B,1)
    -> (B,D)
    """
    B = P.shape[0]

    if not torch.is_tensor(t):
        t = torch.tensor(t, device=P.device, dtype=P.dtype)

    if t.ndim == 0:
        t = t.view(1, 1).expand(B, 1)
    elif t.ndim == 1:
        t = t.view(B, 1)
    else:
        t = t.view(B, 1)

    Q = P
    while Q.shape[1] > 1:
        Q = lerp(Q[:, :-1], Q[:, 1:], t.unsqueeze(-1))
    return Q[:, 0]


def sample_curve(P, N):
    """
    P: (B,K,D) -> (B,N,D)
    """
    ts = torch.linspace(0, 1, N, device=P.device)
    pts = []
    for ti in ts:
        pts.append(de_casteljau(P, ti))  # scalar ti
    return torch.stack(pts, dim=1)


def chamfer(A, B):
    """
    A: (B,NA,D), B: (B,NB,D) -> scalar
    """
    d2 = ((A.unsqueeze(2) - B.unsqueeze(1))**2).sum(-1)
    return d2.min(2).values.mean() + d2.min(1).values.mean()


# ============================
# Model
# ============================

class Model(nn.Module):
    def __init__(self, D, M, hidden=128):
        super().__init__()
        self.M = M
        self.D = D

        self.enc = nn.Sequential(
            nn.Linear(D, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )

        self.decP = nn.Linear(hidden, M * D)
        self.decK = nn.Linear(hidden, M - 1)  # K=2..M

    def forward(self, Qn):
        feat = self.enc(Qn)       # (B,N,H)
        pooled = feat.mean(1)     # (B,H)
        P_all = self.decP(pooled).view(-1, self.M, self.D)
        logitsK = self.decK(pooled)
        return P_all, logitsK


# ============================
# Helpers
# ============================

def sample_pi_st(logits, tau):
    return F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)


def curve_from_pi(P_all, pi, Ncurve):
    """
    P_all: (B,M,D)
    pi: (B,M-1) over K=2..M
    -> (B,Ncurve,D)
    """
    curves = []
    for idx, K in enumerate(range(2, P_all.shape[1] + 1)):
        Pk = P_all[:, :K]
        Ck = sample_curve(Pk, Ncurve)
        curves.append(Ck * pi[:, idx].view(-1, 1, 1))
    return torch.stack(curves).sum(0)


def loss_fn(Qn, P_all, pi):
    C = curve_from_pi(P_all, pi, NCURVE)
    recon = chamfer(C, Qn)
    Ks = torch.arange(2, P_all.shape[1] + 1, device=Qn.device).float()
    expectedK = (pi * Ks.view(1, -1)).sum(1).mean()
    return recon + LAMBDA_SIMPLE * expectedK


def tau_schedule(epoch, max_epoch):
    p = epoch / max_epoch
    return 1.0 * (1 - p) + 0.35 * p


# ============================
# Evaluation
# ============================

@torch.no_grad()
def evaluate(model, Q, K_true, device):
    model.eval()

    Q = Q.to(device)
    K_true = K_true.to(device)

    Qn = normalize_points(Q)
    P_all, logitsK = model(Qn)

    # argmax K (true inference behavior)
    K_hat = logitsK.argmax(-1) + 2
    pi = F.one_hot(K_hat - 2, num_classes=model.M - 1).float()

    C = curve_from_pi(P_all, pi, NCURVE)
    recon = chamfer(C, Qn).item()
    k_acc = (K_hat == K_true).float().mean().item()

    return recon, k_acc


# ============================
# Main


def main():
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = np.load(DATA_PATH)

    Q_train = torch.tensor(data["Q_train"], dtype=torch.float32)
    K_train = torch.tensor(data["K_train"], dtype=torch.long)

    Q_test = torch.tensor(data["Q_test"], dtype=torch.float32)
    K_test = torch.tensor(data["K_test"], dtype=torch.long)

    D = int(data["D"])
    M = int(data["M"])

    model = Model(D, M).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    print("Training...")

    for epoch in range(EPOCHS):
        model.train()
        tau = tau_schedule(epoch, EPOCHS)

        for _ in range(STEPS_PER_EPOCH):
            idx = torch.randint(0, Q_train.shape[0], (BATCH_SIZE,))
            Qb = Q_train[idx].to(device)

            Qn = normalize_points(Qb)
            P_all, logitsK = model(Qn)
            pi = sample_pi_st(logitsK, tau)

            loss = loss_fn(Qn, P_all, pi)

            opt.zero_grad()
            loss.backward()
            opt.step()

        recon, k_acc = evaluate(model, Q_test, K_test, device)
        print(f"epoch {epoch+1:02d} | tau {tau:.3f} | test_recon {recon:.5f} | K_acc {k_acc:.3f}")

    torch.save(model.state_dict(), "bezier_varK_model.pt")
    print("Saved: bezier_varK_model.pt")


if __name__ == "__main__":
    main()