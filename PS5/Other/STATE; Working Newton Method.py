#%% Importing modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import identity
import camb


# %% QUESTION I: Extracting the data and defining the various parameters

# Opening the data file
dataFile = np.loadtxt('./mcmc/COM_PowerSpect_CMB-TT-full_R3.01.txt', skiprows=1)
binnedDataFile = np.loadtxt('./mcmc/COM_PowerSpect_CMB-TT-binned_R3.01.txt', skiprows=1)

# Defining the data columns
multipole = dataFile[:,0]
multipoleVariance = dataFile[:,1]
dataPointError = (dataFile[:,2]+dataFile[:,3])/2


# %% Defining the function which gets the power spectrum, based on Jon's code

def getSpectrum(parameters,maximumHarmonic=3000):
    
    # Defining the relevant model parameters
    hubbleConstant = parameters[0]
    baryonDensity = parameters[1]
    darkMatterDensity = parameters[2]
    opticalDepth = parameters[3]
    primordialAmplitude = parameters[4]
    primordialTilt = parameters[5]
    
    # Initializing paramaters via CAMB
    parameters=camb.CAMBparams()
    parameters.set_cosmology(
        H0=hubbleConstant,ombh2=baryonDensity,omch2=darkMatterDensity,mnu=0.06,omk=0,tau=opticalDepth
        )
    parameters.InitPower.set_params(As=primordialAmplitude,ns=primordialTilt,r=0)
    parameters.set_for_lmax(maximumHarmonic,lens_potential_accuracy=0)
    
    # Defining the resulting parameters, power spectrum, CMB amplitude, and the temperature perturbations
    resultingParameter=camb.get_results(parameters)
    powerSpectrum = resultingParameter.get_cmb_power_spectra(parameters,CMB_unit='muK')
    cosmicMicrowaveBackground = powerSpectrum['total']
    temperaturePerturbations = cosmicMicrowaveBackground[:,0]
    
    return temperaturePerturbations[2:]
    

# %% Computing the power spectrum for the give CMB data

# Defining the sample parameters and the model
sampleParameters = np.asarray([60,0.02,0.1,0.05,2.00e-9,1.0])
sampleModel = getSpectrum(sampleParameters)
sampleModel = sampleModel[:len(multipoleVariance)]

# Defining the residuals and chisquare of the fit
modelResiduals = multipoleVariance - sampleModel
chiSquare = np.sum((modelResiduals/dataPointError)**2)

# Defining the errors for the binned data
binnedError = (binnedDataFile[:,2]+binnedDataFile[:,3])/2

# Plotting the model and outputting the chisquare & degrees of freedom (using binned file for plotting)
plt.plot(multipole,sampleModel, color='mediumspringgreen', label='Model')
plt.errorbar(binnedDataFile[:,0],binnedDataFile[:,1], fmt='.', color='darkviolet', label = 'Data points')
plt.xlabel(r'Multipole Moment [$\ell$]')
plt.ylabel(r'Temperature Fluctuations [$\mu K^2$]')
plt.grid()
plt.legend()

print('The chisquare is ', chiSquare, 'for ', len(modelResiduals)-len(sampleParameters), 'degrees of freedom.')


# %% Repeating the above with slightly different sample parameters

# Defining the sample parameters and the model
sampleParameters = np.asarray([69,0.022,0.12,0.06,2.1e-9,0.95])
sampleModel = getSpectrum(sampleParameters)
sampleModel = sampleModel[:len(multipoleVariance)]

# Defining the residuals and chisquare of the fit
modelResiduals = multipoleVariance - sampleModel
chiSquare = np.sum((modelResiduals/dataPointError)**2)

# Defining the errors for the binned data
binnedError = (binnedDataFile[:,2]+binnedDataFile[:,3])/2

# Plotting the model and outputting the chisquare & degrees of freedom (using binned file for plotting)
plt.plot(multipole,sampleModel, color='mediumspringgreen', label='Model')
plt.errorbar(binnedDataFile[:,0],binnedDataFile[:,1], fmt='.', color='darkviolet', label = 'Data points')
plt.xlabel(r'Multipole Moment [$\ell$]')
plt.ylabel(r'Temperature Fluctuations [$\mu K^2$]')
plt.grid()
plt.legend()

print('The chisquare is ', chiSquare, 'for ', len(modelResiduals)-len(sampleParameters), 'degrees of freedom.')


# %% QUESTION II: Defining the numerical derivative and computing the derivative of the parameter functions

# Defining the numerical derivative and the delta parameter
δ = np.sqrt(1e-16)
numericalDerivative = lambda function,t : 1/(2*δ)*(function(t+δ)-function(t-δ))

# Defining the parameter function to fit
def parameterFunction(p):
    
    # Defining the parameters
    parameters = p
    
    # Paramterizing the function (getSpectrum) wrt different parameters
    hubbleParametrization = lambda x1 : getSpectrum(
        np.concatenate((np.array([x1]),parameters[1:6]))
    )[:len(multipoleVariance)]
    baryonParametrization = lambda x2 : getSpectrum(
        np.concatenate((parameters[0:1],np.array([x2]),parameters[2:6]))
    )[:len(multipoleVariance)]
    darkParametrization = lambda x3 : getSpectrum(
        np.concatenate((parameters[0:2],np.array([x3]),parameters[3:6]))
    )[:len(multipoleVariance)]
    
    opticalParametrization = lambda x4 : getSpectrum(
        np.concatenate((parameters[0:3],np.array([x4]),parameters[4:6]))
    )[:len(multipoleVariance)]
    amplitudeParametrization = lambda x5 : getSpectrum(
        np.concatenate((parameters[0:4],np.array([x5]),parameters[5:6]))
    )[:len(multipoleVariance)]
    tiltParametrization = lambda x6 : getSpectrum(
        np.concatenate((parameters[0:5],np.array([x6])))
    )[:len(multipoleVariance)]
    
    # Defining the analytic derivatives
    gradientValues = np.zeros((len(multipoleVariance),6))
    gradientValues[:,0] = numericalDerivative(hubbleParametrization,parameters[0])
    gradientValues[:,1] = numericalDerivative(baryonParametrization,parameters[1])
    gradientValues[:,2] = numericalDerivative(darkParametrization,parameters[2])
    gradientValues[:,3] = numericalDerivative(opticalParametrization,parameters[3])
    gradientValues[:,4] = numericalDerivative(amplitudeParametrization,parameters[4])
    gradientValues[:,5] = numericalDerivative(tiltParametrization,parameters[5])
    
    return gradientValues

# Computing Newton's method for 25 iterations
initialGuess = np.asarray([6.82469208e+01, 2.23644337e-02, 1.17657399e-01, 8.54343212e-02
 ,2.21927359e-09 ,9.73071475e-01])
initialGuessCopy = np.copy(initialGuess)
for iteration in range(25):
    
    # Defining gradf, the model, and r=d-f
    gradientValue = parameterFunction(initialGuessCopy)
    
    residuals = multipoleVariance - sampleModel
    residuals = np.matrix(residuals).transpose()
    gradientValue = np.matrix(gradientValue)
    
    # Computing step length: δm = (gradf^T * gradf)^-1 * (gradf * r)
    δm = np.linalg.pinv(gradientValue.transpose()*gradientValue)*(gradientValue.transpose()*residuals)
    
    # Updating the new paramters based on the step taken above
    for parameter in range(initialGuessCopy.size):
        initialGuessCopy[parameter] += δm[parameter]  

# Applying a radiative correction to the unstable amplitude parameter
initialGuessCopy[4] = initialGuess[4]
    
# Printing the best-fit parameters computed with numerical derivatives
print('Best fit parameters: ', initialGuessCopy)

# Computing the uncertainty on the best fit parameters
inverseCovariance = identity(dataPointError.size)/(dataPointError)
covarianceMatrix = np.linalg.pinv(
    gradientValue.transpose()@inverseCovariance*gradientValue, rcond = 1e-160
    )
print('Best fit parameter uncertainties: ', np.sqrt(np.diag(covarianceMatrix)))


# %% Plotting the fit, computing chisquare, and saving it to a text file

# Defining the sample parameters and the model
sampleParameters = initialGuessCopy
sampleModel = getSpectrum(sampleParameters)
sampleModel = sampleModel[:len(multipoleVariance)]

# Defining the residuals and chisquare of the fit
modelResiduals = multipoleVariance - sampleModel
chiSquare = np.sum((modelResiduals/dataPointError)**2)

# Defining the errors for the binned data
binnedError = (binnedDataFile[:,2]+binnedDataFile[:,3])/2

# Plotting the model and outputting the chisquare & degrees of freedom (using binned file for plotting)
plt.plot(multipole,sampleModel, color='mediumspringgreen', label='Model')
plt.errorbar(binnedDataFile[:,0],binnedDataFile[:,1], fmt='.', color='darkviolet', label = 'Data points')
plt.xlabel(r'Multipole Moment [$\ell$]')
plt.ylabel(r'Temperature Fluctuations [$\mu K^2$]')
plt.grid()
plt.legend()
plt.savefig('PlotI.png', dpi=500)

print('The chisquare is ', chiSquare, 'for ', len(modelResiduals)-len(sampleParameters), 'degrees of freedom.')

# Saving the result in a text file
with open('planck_fit_params.txt', 'w') as f:
    f.write('Parameter                  Parameter Value        Parameter Uncertainty\n')
    f.write('hubbleConstant             {}      {}\n'.format(sampleParameters[0],np.sqrt(np.diag(covarianceMatrix))[0]))
    f.write('baryonDensity              {}      {}\n'.format(sampleParameters[1],np.sqrt(np.diag(covarianceMatrix))[1]))
    f.write('darkMatterDensity          {}      {}\n'.format(sampleParameters[2],np.sqrt(np.diag(covarianceMatrix))[2]))
    f.write('opticalDepth               {}      {}\n'.format(sampleParameters[3],np.sqrt(np.diag(covarianceMatrix))[3]))
    f.write('primordialAmplitude        {}      {}\n'.format(sampleParameters[4],np.sqrt(np.diag(covarianceMatrix))[4]))
    f.write('primordialTilt             {}      {}\n'.format(sampleParameters[5],np.sqrt(np.diag(covarianceMatrix))[5]))


# %%
