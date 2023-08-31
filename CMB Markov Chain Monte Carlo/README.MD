# Cosmological Microwave Background MCMC
###### For the computational physics graduate course taught by [Professor Jonathan Sievers](https://www.physics.mcgill.ca/~sievers/) at [McGill University](https://www.mcgill.ca/).

![alt text](https://github.com/IsolatedSingularity/Cosmology-Simulations/blob/main/CMB%20Markov%20Chain%20Monte%20Carlo/Plots/PlotIII.png)

## Objective

The cosmic microwave background (CMB) created in the early universe leaves a lasting signal that can be extracted from a noisy background with the use of Markov chain Monte Carlo (MCMC) simulations.

## Code Functionality

The code developed for this assignment is a custom matched-filter program designed to analyze LIGO data for the detection of gravitational waves. It accesses LIGO data from a specified directory, where the unzipped data is stored. The code focuses on events that are seen by both LIGO detectors in Livingston, Louisiana, and Hanford, Washington. It starts by creating separate noise models for the Livingston and Hanford detectors, smoothing the power spectrum, handling any lines present in the data, and applying windowing techniques to the data to avoid tapering where the signal is significant. Using these noise models, the code searches for events listed in the provided JSON file, employing a matched filter approach to identify potential gravitational wave signals. For each event, the code estimates the noise, calculates the signal-to-noise ratio individually for each detector and combined for Livingston and Hanford, and compares these ratios to expected analytic values from the noise model. It determines the frequency where half the signal weight is above and half below, based on the template and noise model, and assesses how accurately it can localize the time of arrival, which is crucial for determining the position of gravitational wave events given the detectors' significant separation.
