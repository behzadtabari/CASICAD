"""
make_bezier_dataset.py

Creates train/test datasets for variable-K Bézier inference.

Each sample contains:
- Q:     (N,D) points sampled on the curve (optionally shuffled, noisy)
- K:     int in [K_min..K_max]
- P_pad: (M,D) padded control points (first K are real, rest are 0)  [for debugging/eval only]
- mask:  (M,)  1 for active control points, 0 otherwise              [for debugging/eval only]

Saves: dataset_bezier_varK_D{D}_N{N}_M{M}.npz

Run:
  python make_bezier_dataset.py
"""

import numpy as np
import torch
from typing import Dict


# -------- geometry (any K) --------
def lerp(a, b, t):
    return (1 - t) * a + t * b


def de_casteljau(P: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    P: (K,D) or (1,K,D)
    t: scalar tensor
    -> (1,D)
    """
    if P.ndim == 2:
        P = P.unsqueeze(0)  # (1,K,D)
    Q = P
    while Q.shape[1] > 1:
        Q = lerp(Q[:, :-1, :], Q[:, 1:, :], t.view(1, 1, 1))
    return Q[:, 0, :]  # (1,D)


def sample_curve_points(P: torch.Tensor, N: int) -> torch.Tensor:
    """
    P: (K,D)
    -> (N,D)
    """
    t_grid = torch.linspace(0.0, 1.0, N, device=P.device)
    pts = []
    for ti in t_grid:
        pts.append(de_casteljau(P, ti)[0])
    return torch.stack(pts, dim=0)


# -------- dataset generation --------
def generate_dataset(
    n_samples: int,
    N: int,
    D: int,
    M: int,
    K_min: int,
    K_max: int,
    noise_std: float,
    shuffle_points: bool,
    seed: int,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    Q_all = np.zeros((n_samples, N, D), dtype=np.float32)
    K_all = np.zeros((n_samples,), dtype=np.int64)
    P_pad_all = np.zeros((n_samples, M, D), dtype=np.float32)
    mask_all = np.zeros((n_samples, M), dtype=np.float32)

    for i in range(n_samples):
        K = int(rng.integers(K_min, K_max + 1))
        P = rng.uniform(-1.5, 1.5, size=(K, D)).astype(np.float32)

        P_t = torch.tensor(P, dtype=torch.float32)
        Q = sample_curve_points(P_t, N).cpu().numpy()

        if noise_std > 0:
            Q = Q + rng.normal(0.0, noise_std, size=Q.shape).astype(np.float32)

        if shuffle_points:
            idx = rng.permutation(N)
            Q = Q[idx]

        P_pad = np.zeros((M, D), dtype=np.float32)
        P_pad[:K] = P

        mask = np.zeros((M,), dtype=np.float32)
        mask[:K] = 1.0

        Q_all[i] = Q
        K_all[i] = K
        P_pad_all[i] = P_pad
        mask_all[i] = mask

    return dict(Q=Q_all, K=K_all, P_pad=P_pad_all, mask=mask_all)


def main():
    # --- config ---
    D = 2
    M = 8
    N = 48
    K_min, K_max = 2, M
    noise_std = 0.01
    shuffle_points = True

    n_train = 50000
    n_test = 5000
    seed = 42

    train = generate_dataset(n_train, N, D, M, K_min, K_max, noise_std, shuffle_points, seed=seed)
    test  = generate_dataset(n_test,  N, D, M, K_min, K_max, noise_std, shuffle_points, seed=seed + 1)

    out_name = f"dataset_bezier_varK_D{D}_N{N}_M{M}.npz"
    np.savez_compressed(
        out_name,
        Q_train=train["Q"], K_train=train["K"], P_pad_train=train["P_pad"], mask_train=train["mask"],
        Q_test=test["Q"],   K_test=test["K"],   P_pad_test=test["P_pad"],   mask_test=test["mask"],
        D=D, N=N, M=M, K_min=K_min, K_max=K_max, noise_std=noise_std, shuffle_points=int(shuffle_points),
    )
    print("Saved:", out_name)
    print("Train Q:", train["Q"].shape, "Test Q:", test["Q"].shape)


if __name__ == "__main__":
    main()