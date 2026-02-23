# Chapter 4 The Bernstein Form of a Bézier Curve


---
## Derivative of The Bezier
This project implements and benchmarks two mathematically equivalent methods for computing the **first derivative** of a Bézier curve using the de Casteljau algorithm.

The goal is to compare their computational performance and ensure that future geometric algorithms remain efficient and scalable.


### Mathematical Background

Let a Bézier curve of degree `n` be defined by control points:

b₀, b₁, …, bₙ

We are interested in computing the first derivative:

b'(t)

---

### Equation (4.26) — Derivative as a Bézier Curve of Differences

b'(t) = n * Σ (b_{j+1} - b_j) * B_j^{n-1}(t)

### Interpretation

1. Compute first differences of control points:

   Δb_j = b_{j+1} - b_j

2. These difference vectors define a new Bézier curve of degree (n−1).
3. Evaluate that curve at parameter t.
4. Multiply the result by n.

### Concept

This method explicitly constructs the derivative curve and evaluates it like a normal Bézier curve.

---

### Equation (4.28) — Derivative from de Casteljau Intermediate Points

b'(t) = n * (b₁^{n−1}(t) − b₀^{n−1}(t))

#### Interpretation

1. Run the standard de Casteljau algorithm to evaluate the curve.
2. When only two points remain (level n−1), store them:

   b₀^{n−1}(t),  b₁^{n−1}(t)

3. The derivative is simply:

   n × (their difference)

#### Concept

The derivative lives one level before the final point in the de Casteljau pyramid.

This method extracts the derivative directly during curve evaluation without constructing a separate Bézier curve.

---

#### Why Benchmark Both?

Although both equations are mathematically identical, their computational behavior may differ:

- Equation (4.26) requires evaluating a separate Bézier curve of degree (n−1).
- Equation (4.28) extracts the derivative during the original de Casteljau evaluation.

Benchmarking helps determine:

- Which implementation is faster in practice
- How both approaches scale with increasing curve degree
- Which method is better suited for future geometric and CAD-related algorithms

Since derivative computation is fundamental in:

- Tangent evaluation
- Curvature estimation
- Surface normals
- CAM toolpath generation

ensuring efficiency now supports robust performance in more advanced geometric pipelines later.


---
## Cusp Regularization - Exercise 4.1
Consider the cubic Bézier curve with planar control points:

b0 = ( 1, 0 )
b1 = (−1, 1 )
b2 = ( 1, 1 )
b3 = (−1, 0 )

At t = 1/2, this curve has a cusp: its first derivative vanishes and it shows a sharp corner.
Now perturb the x-coordinates of b1 and b2 by opposite amounts (keeping symmetry).
Discuss what happens to the curve.

I thought giving it a cool name, Cusp Detox, or Cusp removal, then I thought that how this could be similar to regularization techinques, like enforcing some Lipschitz bounds for the derivates, here we can either smoothe the curve or create a loop outward.  
Run the following script to reproduce the experiment:

```bash
python exc1.py
```

---

