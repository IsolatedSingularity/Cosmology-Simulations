# LIGO Matched-Filter Statistics
###### For the computational physics graduate course taught by [Professor Jonathan Sievers](https://www.physics.mcgill.ca/~sievers/) at [McGill University](https://www.mcgill.ca/).

![alt text](https://github.com/IsolatedSingularity/Cosmology-Simulations/blob/main/Ligo%20Matched%20Filter/LIGOChainsPS6.png)

## Objective

The merging events of black holes creates gravitational perturbations that propagate in our spacetime, known as **gravitational waves**. The signal of these gravitational waves is embedded within primordial noise from perturbations in the early universe. We look at extracting the signal admist the noise using our own matched-filter statistical algorithms for LIGO gravitational wave data.

## Code Functionality

The code developed for this assignment is a efficient MCMC-based tool for analyzing cosmic microwave background (CMB) data. It acquires CMB intensity data from the Planck satellite, focusing on understanding the fundamental parameters governing our universe. At its core, the code employs the CAMB module to generate model CMB power spectra based on various cosmological parameters. It systematically adjusts parameters like the Hubble constant, baryonic matter density, and dark matter density to achieve the best possible match between the model spectra and observed CMB data.

The code evaluates the goodness of fit using the χ² statistic, enabling it to quantify how well the model aligns with the real-world data. It goes further by utilizing numerical optimization techniques to fine-tune these parameters, revealing crucial aspects of our universe, including the Hubble constant.
