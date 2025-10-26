# Self-induced stochastic resonance: A physics-informed machine learning approach

This repo contains the source code for the paper: Self-induced stochastic resonance: A physics-informed machine learning approach

Authors: Divyesh Savaliya, Marius E. Yamakou

Paper link:

Abstract: Self-induced stochastic resonance (SISR) is the emergence of coherent oscillations in slow-fast excitable systems driven solely by noise, without external periodic forcing or proximity to a bifurcation. This work presents a physics-informed machine learning framework for modeling and predicting SISR in the stochastic FitzHugh–Nagumo neuron. We embed the governing stochastic differential equations and SISR-asymptotic timescale-matching constraints directly into a Physics-Informed Neural Network (PINN) based on a Noise-Augmented State Predictor architecture. The composite loss integrates data fidelity, dynamical residuals, and barrier-based physical constraints derived from Kramers’ escape theory. 
The trained PINN accurately predicts the dependence of spike-train coherence on noise intensity, excitability, and timescale separation, matching results from direct stochastic simulations with substantial improvements in accuracy and generalization compared with purely data-driven methods, while requiring significantly less computation. The framework provides a data-efficient and interpretable surrogate model for simulating and analyzing noise-induced coherence in multiscale stochastic systems.

## Prerequisites

- Python version 3.8 to 3.11



