# Custom Random Number Generator
###### For the computational physics graduate course taught by [Professor Jonathan Sievers](https://www.physics.mcgill.ca/~sievers/) at [McGill University](https://www.mcgill.ca/).

![alt text](https://github.com/IsolatedSingularity/Cosmology-Simulations/blob/main/Random%20Number%20Lorentzian/Plots/PlotII.png)

## Objective

We create a custom random number generator to see the non-perturbative effects it has on our code. This is applied to Lorentzian functionals.

## Code Functionality

Problem 1 focuses on demonstrating the inherent flaws in certain pseudo-random number generators (PRNGs) by analyzing the distribution of (x, y, z) triples generated using the C standard library's PRNG. It involves loading and processing data, highlighting the non-random nature of the generated points by visualizing them as lying along planes in 3D space. The code also explores whether Python's random number generator exhibits similar behavior.

Problem 2 deals with generating exponential deviates through a rejection method. It involves writing code to generate exponential deviates from another distribution, such as Lorentzians, Gaussians, or power laws, that can be used for bounding. The code creates a histogram of the generated deviates and compares it to the expected exponential curve, assessing the efficiency of the generator in terms of the fraction of uniform deviates that yield exponential deviates.

Problem 3 extends the rejection method approach, this time utilizing a ratio-of-uniforms generator. The code establishes the limits on the variable v as u ranges from 0 to 1 and evaluates the generator's efficiency in terms of the number of exponential deviates produced per uniform deviate. Similar to Problem 2, it creates a histogram to validate the generated exponential deviates.
