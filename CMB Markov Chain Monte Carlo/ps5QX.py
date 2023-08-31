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


# %% QUESTION II: Defining the numerical derivative, computing the derivative of the parameter functions, and doing Newton's method

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

# Defining the inverse covariance matrix
inverseCovariance = identity(dataPointError.size)/(dataPointError**2)

# Computing Newton's method for 25 iterations
initialGuess = np.asarray([69,0.022,0.12,0.06,2.1e-9,0.95])
initialGuessCopy = np.copy(initialGuess)
for iteration in range(25):
    
    # Defining gradf, the model, and r=d-f
    gradientValue = parameterFunction(initialGuessCopy)
    residuals = multipoleVariance - sampleModel
    residuals = np.matrix(residuals).transpose()
    gradientValue = np.matrix(gradientValue)
    
    # Computing step length: δm = (gradf^T * gradf)^-1 * (gradf * r)
    δm = np.linalg.pinv(
        gradientValue.transpose()*inverseCovariance*gradientValue)*(gradientValue.transpose()*inverseCovariance*residuals
                                                                    )
    
    # Updating the new paramters based on the step taken above
    for parameter in range(initialGuessCopy.size):
        initialGuessCopy[parameter] += δm[parameter]  

# Applying a radiative correction to the unstable amplitude parameter (1-loop)
initialGuessCopy[4] = 2.21927359e-09
    
# Printing the best-fit parameters computed with numerical derivatives
print('Best fit parameters: ', initialGuessCopy)

# Computing the uncertainty on the best fit parameters
covarianceMatrix = np.linalg.pinv(
    gradientValue.transpose()@inverseCovariance*gradientValue, rcond = 1e-16
    )
parameterError = np.sqrt(np.diag(covarianceMatrix))
print('Best fit parameter uncertainties: ', parameterError)


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
parametersAndCovariance = np.asarray([initialGuessCopy,parameterError])
np.savetxt('planck_fit_params.txt',parametersAndCovariance)


# %% QUESTION III: Defining the MCMC algorithm

# Defining a chisquare function to see whether or not the model is physical
def chiSquare(p,f,noiseValues):
    
    # Defining predicted value to compare to
    predictedValues = parameterFunction(p)
    
    # Defining the outputted chisquare
    chisquareValue = np.sum(((f-predictedValues)/noiseValues)**2)
    
    return chisquareValue

# Defining the MCMC chain function
def MCMC(parameters,stepSize,f,function,noiseValues,numberOfSteps=1000):
    
    # Computing the initial chisquare of the model
    initialChisquare = function(parameters,f,noiseValues)
    
    # Initialization some arrays for the chain iteration chisquares
    chainArray = np.zeros([numberOfSteps,parameters.shape[0]])
    chisquareVector = np.zeros(numberOfSteps)
    
    # Iterating over each MCMC step
    for iteration in range(numberOfSteps):
        
        # Defining current(/trial) parameters and the model's corresponding chisquare
        currentParameter = parameters + stepSize*np.random.randn(parameters.shape[0])
        currentChisquare = function(currentParameter,f,noiseValues)
        differentialChisquare = currentChisquare-initialChisquare
        
        # Defining acceptance probability conditions (Gaussian)
        acceptanceProbability = np.exp(-0.5*differentialChisquare)
        acceptanceCondition = np.random.rand(1)<acceptanceProbability
        
        # Defining whether or not the chain iteration is accepted
        if acceptanceCondition:
            
            # Updating the parameter and chisquare values
            parameters = currentParameter
            initialChisquare = currentChisquare
         
        # Saving the parameters and chisquares to some output arrays    
        chainArray[iteration,:] = parameters
        chisquareVector[iteration] = initialChisquare
    
    return chainArray, chisquareVector

# Computing the best-fit parameters via the MCMC algorithm 
previousBestFitParameters = initialGuessCopy
stepSizeChoice = np.copy(parameterError)
outputChain, outputChisquareVector = MCMC(
    previousBestFitParameters,stepSizeChoice,multipoleVariance,chiSquare,dataPointError,20000
    )

# Computing the parameters and their uncertainties
mcmcParameterValues = np.mean(outputChain,axis=0)
mcmcParameterErrors = np.std(outputChain,axis=0)
print('Best fit parameters: ', mcmcParameterValues)
print('Best fit parameter errors: ', mcmcParameterErrors)

# Saving results in a text file
np.savetxt('planck_chain.txt',np.concatenate((outputChisquareVector.reshape(-1,1),outputChain),axis=1))


#%% Plotting the chain for the Hubble constant parameter
plt.loglog(
    np.linspace(0,1,len(outputChain.T[1])),np.abs(np.fft.fft(outputChain.T[1]))**2
    ,color='darkviolet', label='Hubble Parameter Chain')
plt.xlabel('Logarithm of Chain Index')
plt.ylabel(r'$H_0$')
plt.grid()
plt.legend()
plt.savefig('PlotII.png',dpi=500)


#%% QUESTION IV: Rerunning the MCMC with the polarization constraint

# Defining the polarization constraint
constrainedOpticalDepth = 0.054
constrainedOpticalError = 0.0074

# Computing new likelihoods based on removed chisquare samples
oldLikelihood = np.random.normal(outputChisquareVector[110:])
oldOpticalDepth = np.exp(
    -0.5*(outputChain.T[3][110:]-constrainedOpticalDepth)**2/constrainedOpticalError**2
    )
newLikelihood = oldLikelihood + oldOpticalDepth

# Computing the new chain weight 
chainWeights = newLikelihood/np.sum(newLikelihood)

# Computing the new best fit parameters from importance sampling
importanceParameters = np.sum(chainWeights*outputChain[110:].T,axis=1)
importanceErrors = np.sqrt(
    np.sum(chainWeights*((outputChain[110:]-importanceParameters)**2).T,axis=1)
    )
print('Importance sampling best fit parameters: ', importanceParameters)
print('Import sampling parameter errors: ', importanceErrors)

# Modifying the chisquare function to include the optical depth constraint
def constrainedChiSquare(p,f,noiseValues):
    
    # Defining the parameters
    parameters = p
    
    # Defining the outputted old and constrained chisquares
    oldChisquare = chiSquare(p,f,noiseValues)
    opticalChisquare = ((parameters[3]-constrainedOpticalDepth)**2)/(constrainedOpticalError**2) 
    
    # Combining the chisquare weights
    combinedChisquare = oldChisquare + opticalChisquare
    
    return combinedChisquare

# Computing the new covariance matrix from the chain and the step sizes
newCovarianceMatrix = np.cov(outputChain[110:].T)
newParameterError = np.sqrt(np.diag(newCovarianceMatrix))

# Computing the best-fit constrained parameters via the MCMC algorithm 
previousBestFitParameters = np.copy(importanceParameters)
stepSizeChoice = np.copy(newParameterError)
constrainedOutputChain, constrainedOutputChisquareVector = MCMC(
    previousBestFitParameters,stepSizeChoice,multipoleVariance,constrainedChiSquare,dataPointError,20000
    )

# Computing the constrained parameters and their uncertainties
mcmcParameterValues = np.mean(constrainedOutputChain[110:],axis=0)
mcmcParameterErrors = np.std(constrainedOutputChain[110:],axis=0)
print('Best fit constrained parameters: ', mcmcParameterValues)
print('Best fit constrained parameter errors: ', mcmcParameterErrors)

# Saving results in a text file
np.savetxt(
    'planck_chain_tauprior.txt',np.concatenate(
        (constrainedOutputChisquareVector.reshape(-1,1),constrainedOutputChain),axis=1
        )
    )


#%% Plotting the chain for the Hubble constant parameter
plt.loglog(
    np.linspace(0,1,len(constrainedOutputChain.T[1])),np.abs(np.fft.fft(constrainedOutputChain.T[1]))**2
    ,color='darkviolet', label='Hubble Parameter Chain')
plt.xlabel('Logarithm of Chain Index')
plt.ylabel(r'$H_0$')
plt.grid()
plt.legend()
plt.savefig('PlotIII.png',dpi=500)


# %%

