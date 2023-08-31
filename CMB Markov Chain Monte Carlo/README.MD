# Cosmological Microwave Background MCMC
###### For the computational physics graduate course taught by [Professor Jonathan Sievers](https://www.physics.mcgill.ca/~sievers/) at [McGill University](https://www.mcgill.ca/).

![alt text](https://github.com/IsolatedSingularity/Cosmology-Simulations/blob/main/CMB%20Markov%20Chain%20Monte%20Carlo/Plots/PlotIII.png)

## Objective

Phase transitions in the very early universe cause spontaneous symmetry breaking events which create topological defects. Linear topological defects associated with the U(1) symmetry group are known as **cosmic strings** and create high energy signals in the form of wakes as they propagate through spacetime. Their presence thus is of interest to study the high energy structure of the universe and the standard model.

The strings occur in a class of renormalizable quantum field theories and are stable under specific conditions on the spacetime manifold's homotopy groups. Their dynamics are given by the Nambu-Gotto action, much like bosonic excitations in string theory. What's more is that gravitational backreaction during the early universe causes primordial ΛCDM noise which hides the string signal. Thus the purpose of this repository is to develop statistics to efficiently extract the cosmic string signal admist the non-linear noise through the framework of 21cm cosmology.

## Code Functionality

*Cosmic String Extraction Statistics.py* builds the cosmic string signal from scratch as a finite density of energy radiating a certain temperature difference admist the 21cm background temperature map. This propagates through spacetime and traces out a wake which has the temperature gradient defined on its convex hull. Then using universe simulations done with 21cmFAST, the string signal is embedded in the primordial noise. Finally, to extract the dynamic signal we make use of statistics such as correlation functions, matched filters, and wavelets. The output are plots of these statistics when the signal of the string is detected in the noise.

The report covering all the theory and code can be found in the main repository as a PDF file.
