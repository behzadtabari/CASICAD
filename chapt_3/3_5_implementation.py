j = 0
def decas_3d(coeff, t):
    """
    Recursive de Casteljau algorithm in 3D.

    coeff : list of control points [(x,y,z), (x,y,z), ...]
    t     : parameter value (0 <= t <= 1)

    Returns: (x, y, z) point on the Bézier curve at parameter t
    """
    global j
    n = len(coeff)
    if n == 1:  # base case
        return coeff[0]

    # interpolate between consecutive 3D points
    next_level = [
        (
            (1 - t) * coeff[i][0] + t * coeff[i + 1][0],  # x
            (1 - t) * coeff[i][1] + t * coeff[i + 1][1],  # y
            (1 - t) * coeff[i][2] + t * coeff[i + 1][2],  # z
        )
        for i in range(n - 1)
    ]
    print(f"calling the algorithm {j} times")
    j = j + 1
    print(next_level)
    return decas_3d(next_level, t)

# Define 3D control points
control_points = [(0, 0, 0), (1, 2, 1), (3, 3, -1), (4, 0, 0)]

# Evaluate curve at a few t values
print(decas_3d(control_points, 0.0))   # start point -> (0,0,0)
'''
print(decas_3d(control_points, 0.5))   # midpoint (on the curve)
print(decas_3d(control_points, 1.0))   # end point -> (4,0,0)
'''