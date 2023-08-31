# Custom Integrator for Analytic Functionals
###### For the computational physics graduate course taught by [Professor Jonathan Sievers](https://www.physics.mcgill.ca/~sievers/) at [McGill University](https://www.mcgill.ca/).

![alt text](https://github.com/IsolatedSingularity/Cosmology-Simulations/blob/main/Chebyshev%20Functional%20Expansion/Plots/PlotIV.png)

## Objective

We create a custom integrator to effiently compute solutions to non-transcendental PDEs.

## Code Functionality

Problem 1: It implements an RK4 integrator with rk4_step and rk4_stepd functions. The rk4_step function takes a step of length h to numerically integrate the given ODE. The rk4_stepd function improves accuracy by taking steps of length h and comparing them to two steps of length h/2 to reduce leading-order errors.

Problem 2: The code solves the decay chain problem of U238 using an appropriate ODE solver from SciPy, considering all decay products in the chain. It then plots the ratio of Pb206 to U238 and the ratio of Thorium 230 to U234 over a relevant time range for analytical interpretation.

Problem 3: The code performs a linear least-squares fit to real photogrammetry data from the file dish zenith.txt. It fits a rotationally symmetric paraboloid to the data, transforming the nonlinear problem into a linear one by introducing new parameters. The code then carries out the fit, determining the best-fit parameters for the paraboloid. Additionally, it estimates the noise in the data and uses it to calculate the uncertainty in the focal length, comparing it to the target focal length of 1.5 meters.
