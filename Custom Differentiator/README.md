# Custom Functional Differentiator
###### For the computational physics graduate course taught by [Professor Jonathan Sievers](https://www.physics.mcgill.ca/~sievers/) at [McGill University](https://www.mcgill.ca/).

![alt text](https://github.com/IsolatedSingularity/Cosmology-Simulations/blob/main/Custom%20Differentiator/Plots/cosineDifference.png)

## Objective

We create a custom differentiator to effiently compute expansions of analytic functionals.

## Code Functionality
Problem 1, it calculates the numerical derivative of a given function f(x) at a specific point x, estimating the optimal step size δx based on function properties and machine precision, and validates this approach with f(x) = exp(x) and f(x) = exp(0.01x) functions. In Problem 2, the code defines a ndiff function that computes the derivative at a designated point, optionally returning the derivative, δx, and an error estimate using the centered difference formula and optimal δx calculation. Problem 3 involves creating a routine for interpolating temperatures from voltage data using Lakeshore 670 diode information, estimating uncertainty in temperature values. Finally, Problem 4 compares the accuracy of different interpolation methods (polynomial, cubic spline, rational function) when applied to cos(x) and the Lorentzian function 1/(1 + x^2) over specific intervals. The code assesses error in approximating the Lorentzian function with rational functions, explores the impact of increasing rational function orders, and investigates the effects of using np.linalg.inv versus np.linalg.pinv for coefficient calculations, analyzing changes in p and q.
