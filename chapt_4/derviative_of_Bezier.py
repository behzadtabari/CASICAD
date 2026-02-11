import numpy as np
import matplotlib.pyplot as plt
import time


# -----------------------------
# Core: de Casteljau utilities
# -----------------------------
def de_casteljau_point(P, t):
    """
    Evaluate a Bézier curve at parameter t using iterative de Casteljau.

    P : list/array of control points shape (m, 3)
    t : float in [0, 1]
    """
    Q = np.array(P, dtype=float).copy()
    n = len(Q)
    for k in range(1, n):
        Q[: n - k] = (1 - t) * Q[: n - k] + t * Q[1 : n - k + 1]
    return Q[0]


def de_casteljau_point_and_pair(P, t):
    """
    Evaluate Bézier curve at t and ALSO return the two points at level (n-1),
    i.e., when the de Casteljau pyramid has exactly 2 points left.

    This pair (b0^{n-1}(t), b1^{n-1}(t)) is exactly what Eq. 4.28 (r=1) needs.

    Returns:
      point : b(t)  (shape (3,))
      pair  : (p0, p1) each shape (3,)
    """
    Q = np.array(P, dtype=float).copy()
    n = len(Q)

    pair = None
    for k in range(1, n):
        Q[: n - k] = (1 - t) * Q[: n - k] + t * Q[1 : n - k + 1]
        if n - k == 2:  # exactly two points remain -> level (n-1)
            pair = (Q[0].copy(), Q[1].copy())

    return Q[0], pair


# -----------------------------
# Eq. 4.26 and Eq. 4.28 (r=1)
# -----------------------------
def deriv_eq_426_r1(P, t):
    """
    Eq. 4.26 with r=1:
      b'(t) = n * sum_{j=0}^{n-1} (b_{j+1}-b_j) * B_j^{n-1}(t)

    Implementation:
      1) Compute first differences Δb_j = b_{j+1}-b_j (control points of derivative curve)
      2) Evaluate that (degree n-1) Bézier curve at t using de Casteljau
      3) Multiply by n
    """
    P = np.array(P, dtype=float)
    n = len(P) - 1
    dP = P[1:] - P[:-1]              # Δ b_j
    return n * de_casteljau_point(dP, t)


def deriv_eq_428_r1(P, t):
    """
    Eq. 4.28 with r=1:
      b'(t) = n * (b_1^{n-1}(t) - b_0^{n-1}(t))

    Implementation:
      Run one de Casteljau evaluation and capture the final pair when 2 points remain.
    """
    n = len(P) - 1
    _, (p0, p1) = de_casteljau_point_and_pair(P, t)
    return n * (p1 - p0)


# -----------------------------
# Sampling + plotting
# -----------------------------
def sample_curve(P, num=200):
    ts = np.linspace(0.0, 1.0, num)
    pts = np.array([de_casteljau_point(P, t) for t in ts])
    return ts, pts


def sample_derivative(P, num=200, method="426"):
    ts = np.linspace(0.0, 1.0, num)
    if method == "426":
        d = np.array([deriv_eq_426_r1(P, t) for t in ts])
    elif method == "428":
        d = np.array([deriv_eq_428_r1(P, t) for t in ts])
    else:
        raise ValueError("method must be '426' or '428'")
    return ts, d


# -----------------------------
# Benchmarking
# -----------------------------
def benchmark(P, num=200, loops=500):
    """
    Time Eq 4.26 (r=1) vs Eq 4.28 (r=1) over 'num' t-samples repeated 'loops' times.
    """
    ts = np.linspace(0.0, 1.0, num)

    # Warm-up
    for t in ts:
        deriv_eq_426_r1(P, t)
        deriv_eq_428_r1(P, t)

    # Eq 4.26 timing
    t0 = time.perf_counter()
    for _ in range(loops):
        for t in ts:
            deriv_eq_426_r1(P, t)
    t1 = time.perf_counter()

    # Eq 4.28 timing
    t2 = time.perf_counter()
    for _ in range(loops):
        for t in ts:
            deriv_eq_428_r1(P, t)
    t3 = time.perf_counter()

    return (t1 - t0), (t3 - t2)


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    # 5 control points => degree n=4 (quartic Bézier)
    control_points = [(0, 0, 0), (1, 2, 1), (3, 3, -1), (4, 0, 0)]
    P = np.array(control_points, dtype=float)

    # Sample curve
    ts, curve = sample_curve(P, num=200)

    # Optional: sample derivatives (just to verify they look reasonable)
    _, deriv426 = sample_derivative(P, num=200, method="426")
    _, deriv428 = sample_derivative(P, num=200, method="428")

    # Benchmark
    dt426, dt428 = benchmark(P, num=200, loops=500)
    print(f"Eq 4.26 (r=1) time: {dt426:.6f} s")
    print(f"Eq 4.28 (r=1) time: {dt428:.6f} s")

    # Plot curve + control polygon
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(P[:, 0], P[:, 1], P[:, 2], marker="o")      # control polygon
    ax.plot(curve[:, 0], curve[:, 1], curve[:, 2])      # Bézier curve

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Bézier curve (de Casteljau)")

    plt.show()