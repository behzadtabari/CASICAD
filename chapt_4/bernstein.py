# ========================================================
# Author: Behzad Tabari
# Date: 2025-09-14
# Description: This script is about to explain the Bernstein polynomials
# ========================================================
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import math

def bernstein_poly(k, n, x):
    """Compute Bernstein polynomial B_{k,n}(x)."""
    return math.comb(n, k) * (x**k) * ((1 - x)**(n - k))

# Parameters
n = int(input("Enter the number of polynomials: "))
x = np.linspace(0, 1, 500)

# Plot all Bernstein polynomials of degree n
plt.figure(figsize=(8, 6))
for k in range(n + 1):
    y = [bernstein_poly(k, n, xi) for xi in x]
    plt.plot(x, y, label=f"$B_{{{k},{n}}}(x)$")

plt.title(f"Bernstein Polynomials of Degree {n}")
plt.xlabel("x")
plt.ylabel("B_{k,n}(x)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show(block=True)


while True:
    try:
        answer = (input("Do you want to check the partition of unity? (Y/N): ")
                  .strip().upper())
        if answer not in ("Y", "N"):
            raise ValueError("Invalid input, please enter Y or N.")
        if answer == "N":
            break

        t = float(input("Enter t (a float between 0 and 1): "))
        if not (0 <= t <= 1):
            raise ValueError("Invalid input, please enter a value between 0 "
                             "and 1.")

        # compute values
        y = [bernstein_poly(j, n, t) for j in range(n+1)]
        for j, val in enumerate(y):
            print(f"B_{j, n}({t}) = {val}")


        if np.isclose(sum(y), 1.0):
            print("Sum the numbers, of course partition of unity works, "
                  "do not be pedantic about those tiny decimals,"
                  " if you are familiar with it you know probably why, "
                  "if you don't try to find out.")
        else:
            print(" Something went wrong! Debug the code")

        break
    except ValueError as e:
        print(e)







