# Chapter 3: The de Casteljau Algorithm


---
## Exercises


### Exc 2

The exercise body reads as below,

   "Use the de Casteljau algorithm to design a curve of degree four that has its middle control point on the curve."

Well well well, we are living in the world of AI,and inspired by this exercise's text body, I present a Neural Bézier Curve Inference (PINN-like)

This project learns to recover Bézier control points directly from sampled points on a curve.


Instead of solving an optimization problem for each input, a neural network is trained to perform one-shot inference:

set of curve points Q → control points 

Idea

The Bézier curve equation (De Casteljau algorithm) acts as a known forward physics model.

A neural network predicts control points from sampled curve points.

Training uses a physics-based loss: the predicted control points generate a curve that must reconstruct the input points.

### Reproducing the Result

Run the following script to reproduce the experiment:

```bash
python bezier_variable_k_full.py
```

---



