# Chebyshev & Legendre Functional Expansion
###### For the computational physics graduate course taught by [Professor Jonathan Sievers](https://www.physics.mcgill.ca/~sievers/) at [McGill University](https://www.mcgill.ca/).

![alt text](https://github.com/IsolatedSingularity/Cosmology-Simulations/blob/main/Chebyshev%20Functional%20Expansion/Plots/PlotIV.png)

## Objective

We expand analytic functionals as Legendre or Chebyshev sums over the appropriate region over which they are defined on to minimize integrator errors.

## Code Functionality

The code for this assignment would involve solving several physics and numerical integration problems. For Problem 1, it would calculate and plot the electric field from an infinitesimally thin spherical shell of charge as a function of distance from the center of the sphere. This involves integrating the contributions of rings along its central axis to create the electric field distribution. Both a custom integrator and the scipy.integrate.quad function would be used to perform the integration, with attention to regions where z < R and z > R. The code would identify if there is a singularity in the integral and check whether quad or the custom integrator can handle it.

For Problem 2, the code would implement a recursive variable step size integrator, similar to what was discussed in class, but with the added constraint of not calling f(x) multiple times for the same x. This integrator would be defined with a function prototype as described and applied to various examples to measure the reduction in function calls compared to a non-optimized approach.

For Problem 3, the code would create a function modeling the logarithm base 2 of x within a specific range and accuracy using truncated Chebyshev polynomial fitting. It would determine the number of terms needed for the fit and then extend this to write a routine called mylog2 capable of calculating the natural logarithm of any positive number. This involves using np.frexp to break down floating-point numbers and np.polynomial.chebyshev.chebval to evaluate the Chebyshev fit. The code may also include a bonus section comparing the Chebyshev method with Legendre polynomials or Taylor series in terms of RMS and maximum error when calculating logarithms.
