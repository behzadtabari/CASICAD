import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
matplotlib.use("TkAgg")   # or "Qt5Agg"


# Control points
P0 = np.array([0.0, 0.0])
P1 = np.array([1.0, 2.0])
P2 = np.array([3.0, 0.0])

# Quadratic Bézier curve
def bezier_quad(t, P0, P1, P2):
    return (1-t)**2 * P0 + 2*(1-t)*t * P1 + t**2 * P2

# Blossom version: two parameters u,v
def de_casteljau_step(P0, P1, P2, u, v):
    Q0 = (1-u)*P0 + u*P1  # first interpolation with u
    Q1 = (1-u)*P1 + u*P2
    R  = (1-v)*Q0 + v*Q1  # second interpolation with v
    return Q0, Q1, R

# Precompute Bézier curve
ts = np.linspace(0,1,200)
curve = np.array([bezier_quad(t,P0,P1,P2) for t in ts])

# Set up figure
fig, ax = plt.subplots(figsize=(7,5))
ax.plot(curve[:,0], curve[:,1], 'b-', label="Bézier Curve")
ax.plot([P0[0], P1[0], P2[0]], [P0[1], P1[1], P2[1]], 'k--o', label="Control Polygon")
ax.set_aspect("equal")
ax.set_xlim(-0.2, 3.2)
ax.set_ylim(-0.2, 2.2)
ax.legend()

# Lines and points for animation
line_q,  = ax.plot([], [], 'g--o', label="First interp (u)")
line_r,  = ax.plot([], [], 'r--o', label="Second interp (v)")
point_R, = ax.plot([], [], 'ro', markersize=10, label="Blossom b(u,v)")

def init():
    line_q.set_data([], [])
    line_r.set_data([], [])
    point_R.set_data([], [])
    return line_q, line_r, point_R

def animate(frame):
    u = frame/50.0        # u goes 0 → 1
    #v = 1.0 - u           # v goes opposite, 1 → 0
    v = u
    Q0, Q1, R = de_casteljau_step(P0, P1, P2, u, v)

    line_q.set_data([Q0[0], Q1[0]], [Q0[1], Q1[1]])
    line_r.set_data([Q0[0], R[0], Q1[0]], [Q0[1], R[1], Q1[1]])
    point_R.set_data([R[0]], [R[1]])   # <-- FIXED

    return line_q, line_r, point_R

ani = animation.FuncAnimation(fig, animate, frames=51, init_func=init,
                              interval=200, blit=True)
# or "blossom.gif", writer="pillow"
ani.save("blossom.mp4", writer="ffmpeg")

