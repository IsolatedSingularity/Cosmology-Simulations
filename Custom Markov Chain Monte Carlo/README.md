# Custom Markov chain Monte Carlo 
###### For the computational physics graduate course taught by [Professor Jonathan Sievers](https://www.physics.mcgill.ca/~sievers/) at [McGill University](https://www.mcgill.ca/).

![alt text](https://github.com/IsolatedSingularity/Cosmology-Simulations/blob/main/Custom%20Markov%20Chain%20Monte%20Carlo/Plots/PlotIV.png)

## Objective

We create a Markov chain Monte Carlo simulation algorithm to be used in fitting CMB and LIGO signal functionals in non-perturbative noise.

## Code Functionality

In Problem 1, the code loads data from the file sidebands.npz containing optical cavity resonance measurements, models it as a single Lorentzian using analytic derivatives, and employs Newton's method or Levenberg-Marquardt for fitting. It calculates the best-fit parameters for the Lorentzian's amplitude, width, and center and estimates the data's noise to derive parameter errors. Problem 2 replicates the procedure from Problem 1 but uses numerical derivatives, comparing the results with those obtained analytically to gauge their statistical significance. In Problem 3, the code models the data as the sum of three Lorentzian functions, sharing parameters and known separations, and estimates initial guesses and errors for additional parameters. Problem 4 examines residuals by subtracting the best-fit model from the data to assess error assumptions and model completeness. Problem 5 generates realizations of parameter errors based on the covariance matrix from Problem 3 and plots models using these perturbed parameters, computing the difference in the 2 statistic between perturbed and best-fit parameters. Problem 6 implements the Markov Chain Monte Carlo (MCMC) method for data fitting, generating trial samples with the parameter covariance estimate, ensuring convergence through a convergence plot, and investigating changes in error bars due to MCMC. Lastly, in Problem 7, the code calculates the actual cavity resonance width in GHz using the provided information. The code deploys numerical techniques, fitting methods, and statistical analyses to effectively address these problems.
