# ========================================================
# Author: Behzad Tabari
# Date: 2025-09-15
# Description: This script is about Hodographs, these were introduce by a guy
# called William Rowan Hamilton (1846), to understand them imagine you are
# Lewis Hamilton on a Bezier Curve and you also have the velocity curve, I hope
# you get the analogy, I think they could be really useful for assessing the
# continuity between Bezier curves
# ========================================================

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.special import comb

# Evaluate a Bezier curve at parameter values t
def bezier_curve(control_points, t):
    n = len(control_points) - 1
    curve = np.zeros((len(t), control_points.shape[1]))
    for i in range(n + 1):
        bernstein = comb(n, i) * (t**i) * ((1 - t)**(n - i))
        curve += bernstein[:, None] * control_points[i]
    return curve

# Compute control points of the derivative (hodograph)
def bezier_derivative_points(control_points):
    n = len(control_points) - 1
    return n * (control_points[1:] - control_points[:-1])

# Plot original Bezier + derivative hodograph + tangent vectors
def plot_bezier_and_hodograph(control_points, n_samples=200):
    t = np.linspace(0, 1, n_samples)

    # Original curve
    curve = bezier_curve(control_points, t)

    # Derivative control points & hodograph
    deriv_points = bezier_derivative_points(control_points)
    hodograph = bezier_curve(deriv_points, t)

    # Evaluate derivative at sample points
    tangents = bezier_curve(deriv_points, t)

    # Plot
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Left: original curve + tangent vectors
    axs[0].plot(curve[:,0], curve[:,1], 'b-', label="Bezier curve")
    axs[0].plot(control_points[:,0], control_points[:,1], 'ro--', label="Control polygon")

    # Add tangent vectors along the curve (scaled for visibility)
    step = n_samples // 20
    for i in range(0, n_samples, step):
        axs[0].arrow(curve[i,0], curve[i,1],
                     tangents[i,0]*0.1, tangents[i,1]*0.1,
                     head_width=0.05, color="green", alpha=0.7)

    axs[0].set_title("Bezier curve with tangent vectors")
    axs[0].legend()
    axs[0].axis("equal")

    # Right: hodograph (derivative curve)
    axs[1].plot(hodograph[:,0], hodograph[:,1], 'g-', label="Hodograph")
    axs[1].plot(deriv_points[:,0], deriv_points[:,1], 'ms--',
                label="Hodograph control polygon")
    axs[1].set_title("Hodograph (Bezier derivative)")
    axs[1].legend()
    axs[1].axis("equal")

    plt.show()


# Example: Cubic Bezier
P0 = np.array([0.0, 0.0])
P1 = np.array([1.0, 2.0])
P2 = np.array([3.0, 3.0])
P3 = np.array([4.0, 0.0])
control_points = np.array([P0, P1, P2, P3])

plot_bezier_and_hodograph(control_points)
