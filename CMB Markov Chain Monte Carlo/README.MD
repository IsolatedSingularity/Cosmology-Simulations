# Cosmological Microwave Background MCMC
###### For the computational physics graduate course taught by [Professor Jonathan Sievers](https://www.physics.mcgill.ca/~sievers/) at [McGill University](https://www.mcgill.ca/).

![alt text](https://github.com/IsolatedSingularity/Cosmology-Simulations/blob/main/CMB%20Markov%20Chain%20Monte%20Carlo/Plots/PlotIII.png)

## Objective

The cosmic microwave background (CMB) created in the early universe leaves a lasting signal that can be extracted from a noisy background with the use of Markov chain Monte Carlo (MCMC) simulations.

## Code Functionality

The code developed for this assignment is a efficient MCMC-based tool for analyzing cosmic microwave background (CMB) data. It acquires CMB intensity data from the Planck satellite, focusing on understanding the fundamental parameters governing our universe. At its core, the code employs the CAMB module to generate model CMB power spectra based on various cosmological parameters. It systematically adjusts parameters like the Hubble constant, baryonic matter density, and dark matter density to achieve the best possible match between the model spectra and observed CMB data. The code evaluates the goodness of fit using the χ² statistic, enabling it to quantify how well the model aligns with the real-world data. It goes further by utilizing numerical optimization techniques to fine-tune these parameters, revealing crucial aspects of our universe, including the Hubble constant.
