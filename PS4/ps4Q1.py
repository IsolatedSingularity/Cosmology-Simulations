#%% Importing modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import identity


# %% Loading data file
dataFile = np.load('./mcmc/sidebands.npz')
timeRange = dataFile['time']
signalValues = dataFile['signal']


# %% Computing Newton's method w/ analytic derivatives

# Defining Lorentzian fit function and its analytic derivatives
def lorentzianFunction(p,t):
    
    # Defining parameters
    a, t0, omega = p
    
    # Defining the function itself
    functionValue = a/( 1 + ((t-t0)**2)/(omega**2) )
    
    # Defining the analytic derivatives
    gradientValues = np.zeros([t.size,len(p)])
    gradientValues[:,0] = 1/( 1+ ( (t-t0)**2 ) / (omega**2) )
    gradientValues[:,1] = (2*a*(t-t0)*omega**2) / ( ((t-t0)**2+omega**2)**2 )
    gradientValues[:,2] = (2*a*omega*(t-t0)**2) / ( ((t-t0)**2+omega**2)**2 )
    
    return functionValue, gradientValues
      
# Computing Newton's method for 25 iterations
initialGuess = np.array([1.5,2e-4,1e-4]) #GUESS AFFECTS OUTCOME A LOT
initialGuessCopy = np.copy(initialGuess)
for iteration in range(25):
    
    # Defining f, gradf, and r=d-f
    predictedValue, gradientValue = lorentzianFunction(initialGuessCopy, timeRange)
    residuals = signalValues-predictedValue
    residuals = np.matrix(residuals).transpose()
    gradientValue = np.matrix(gradientValue)
    
    
    # Computing step length: δm = (gradf^T * gradf)^-1 * (gradf * r)
    δm = np.linalg.pinv(gradientValue.transpose()*gradientValue)*(gradientValue.transpose()*residuals)
    
    # Updating the new paramters based on the step taken above
    for parameter in range(initialGuessCopy.size):
        initialGuessCopy[parameter] += δm[parameter]
        
# Plotting output of Newton's method & printing the best-fit parameters
plt.plot(timeRange,signalValues,color = 'mediumspringgreen', label='Signal Data')
plt.plot(timeRange,predictedValue, color='darkviolet', label='Best Fit')
plt.xlabel('t')
plt.ylabel('Signal')
plt.grid()
plt.legend()
plt.savefig('PlotI.png',dpi=500)
plt.show()

print('Best fit parameters: ', 'a,t0,omega = ', initialGuessCopy)


# %% Computing the uncertainty on the best fit parameters

# Defining the standard error of each data points
σ = 5e-3

# Computing the covariance matrix inverse, and the covariance matrix
inverseCovariance = identity(timeRange.size)/(σ**2)
covarianceMatrix = np.linalg.pinv(
    gradientValue.transpose()@inverseCovariance*gradientValue
    )

# Printing the uncertainty of the best fit parameters
print('Best fit parameter uncertainties: ', 'σa,σt0,σomega = ', np.sqrt(np.diag(covarianceMatrix)))


# %% Computing Newton's method w/ numerical derivatives

# Defining a numerical derivative that takes input functions
δ = np.sqrt(1e-16)
numericalDerivative = lambda function,t : 1/(2*δ)*(function(t+δ)-function(t-δ))

# Defining Lorentzian fit function and its numerical derivatives
def lorentzianFunction(p,t):
    
    # Defining parameters
    a, t0, omega = p
    
    # Defining the function itself
    functionValue = a/( 1 + ((t-t0)**2)/(omega**2) )
    
    # Parametrizing the function wrt different parameters & keeping t fixed
    aParametrization = lambda x1 : x1/( 1 + ((t-t0)**2)/(omega**2) )
    t0Parametrization = lambda x2 : a/( 1 + ((t-x2)**2)/(omega**2) )
    omegaParametrization =  lambda x3 : a/( 1 + ((t-t0)**2)/(x3**2) )
    
    # Defining the analytic derivatives
    gradientValues = np.zeros([t.size,len(p)])
    gradientValues[:,0] = numericalDerivative(aParametrization,a)
    gradientValues[:,1] = numericalDerivative(t0Parametrization,t0)
    gradientValues[:,2] = numericalDerivative(omegaParametrization,omega)
    
    return functionValue, gradientValues
      
# Computing Newton's method for 25 iterations
initialGuess = np.array([1.5,2e-4,1e-4]) #Note: GUESS AFFECTS OUTCOME A LOT
initialGuessCopy = np.copy(initialGuess)
for iteration in range(25):
    
    # Defining f, gradf, and r=d-f
    predictedValue, gradientValue = lorentzianFunction(initialGuessCopy, timeRange)
    residuals = signalValues-predictedValue
    residuals = np.matrix(residuals).transpose()
    gradientValue = np.matrix(gradientValue)
    
    
    # Computing step length: δm = (gradf^T * gradf)^-1 * (gradf * r)
    δm = np.linalg.pinv(gradientValue.transpose()*gradientValue)*(gradientValue.transpose()*residuals)
    
    # Updating the new paramters based on the step taken above
    for parameter in range(initialGuessCopy.size):
        initialGuessCopy[parameter] += δm[parameter]    

# Printing the best-fit parameters computed with numerical derivatives
print('Best fit parameters: ', 'a,t0,omega = ', initialGuessCopy)


# %% Computing Newton's method w/ numerical derivatives + a triple Lorentzian

def tripleLorentzianFunction(p,t):
    
    # Defining parameters
    a, b, c, t0, omega, dt = p #Note: DIFFERENT ORDER
    
    # Defining the function itself
    functionValue = a/( 1 + ((t-t0)**2)/(omega**2) ) + b/( 1 + ((t-t0+dt)**2)/(omega**2) ) + c/( 1 + ((t-t0-dt)**2)/(omega**2) )
    
    # Parametrizing the function wrt different parameters & keeping t fixed
    aParametrization = lambda x1 : x1/( 1 + ((t-t0)**2)/(omega**2) ) + b/( 1 + ((t-t0+dt)**2)/(omega**2) ) + c/( 1 + ((t-t0-dt)**2)/(omega**2) )
    bParametrization = lambda x2 : a/( 1 + ((t-t0)**2)/(omega**2) ) + x2/( 1 + ((t-t0+dt)**2)/(omega**2) ) + c/( 1 + ((t-t0-dt)**2)/(omega**2) )
    cParametrization = lambda x3 : a/( 1 + ((t-t0)**2)/(omega**2) ) + b/( 1 + ((t-t0+dt)**2)/(omega**2) ) + x3/( 1 + ((t-t0-dt)**2)/(omega**2) )
    t0Parametrization = lambda x4 : a/( 1 + ((t-x4)**2)/(omega**2) ) + b/( 1 + ((t-x4+dt)**2)/(omega**2) ) + c/( 1 + ((t-x4-dt)**2)/(omega**2) )
    omegaParametrization =  lambda x5 : a/( 1 + ((t-t0)**2)/(x5**2) ) + b/( 1 + ((t-t0+dt)**2)/(x5**2) ) + c/( 1 + ((t-t0-dt)**2)/(x5**2) )
    dtParametrization = lambda x6 : a/( 1 + ((t-t0)**2)/(omega**2) ) + b/( 1 + ((t-t0+x6)**2)/(omega**2) ) + c/( 1 + ((t-t0-x6)**2)/(omega**2) )
    
    # Defining the analytic derivatives
    gradientValues = np.zeros([t.size,len(p)])
    gradientValues[:,0] = numericalDerivative(aParametrization,a)
    gradientValues[:,1] = numericalDerivative(bParametrization,b)
    gradientValues[:,2] = numericalDerivative(cParametrization,c)
    gradientValues[:,3] = numericalDerivative(t0Parametrization,t0)
    gradientValues[:,4] = numericalDerivative(omegaParametrization,omega)
    gradientValues[:,5] = numericalDerivative(dtParametrization,dt)
    
    return functionValue, gradientValues

# Computing Newton's method for 25 iterations
initialGuess = np.array([
    1.42,1e-1,1e-1,0.000192,0.000018,0.00004
    ]) #Note: GUESS AFFECTS OUTCOME A LOT
initialGuessCopy = np.copy(initialGuess)
for iteration in range(25):
    
    # Defining f, gradf, and r=d-f
    predictedValue, gradientValue = tripleLorentzianFunction(initialGuessCopy, timeRange)
    residuals = signalValues-predictedValue
    residuals = np.matrix(residuals).transpose()
    gradientValue = np.matrix(gradientValue)
    
    
    # Computing step length: δm = (gradf^T * gradf)^-1 * (gradf * r)
    δm = np.linalg.pinv(gradientValue.transpose()*gradientValue)*(gradientValue.transpose()*residuals)
    
    # Updating the new paramters based on the step taken above
    for parameter in range(initialGuessCopy.size):
        initialGuessCopy[parameter] += δm[parameter]    

# Printing the best-fit parameters computed with numerical derivatives
print('Best fit parameters: ', 'a,b,c,t0,omega,dt = ', initialGuessCopy)

# Plotting output of Newton's method
plt.plot(timeRange,signalValues,color = 'mediumspringgreen', label='Signal Data')
plt.plot(timeRange,predictedValue, color='darkviolet', label='Best Fit')
plt.xlabel('t')
plt.ylabel('Signal')
plt.grid()
plt.legend()
plt.savefig('PlotII.png',dpi=500)
plt.show()

# Computing the uncertainty on the best fit parameters (as before)
inverseCovariance = identity(timeRange.size)/(σ**2)
covarianceMatrix = np.linalg.pinv(
    gradientValue.transpose()@inverseCovariance*gradientValue, rcond = 1e-160
    )

# Printing the uncertainty of the best fit parameters
print('Best fit parameter uncertainties: ', 'σa,σb,σc,σt0,σomega,σdt = ', np.sqrt(np.diag(covarianceMatrix)))


# %% Plotting the residuals of the data
plt.plot(timeRange,signalValues-predictedValue, color = 'darkviolet', lw=0, marker='.')
plt.xlabel('t')
plt.ylabel('Residual')
plt.grid()
plt.legend()
plt.savefig('PlotIII.png',dpi=500)
plt.show()


# %% Computing multiple realizations of the parameter errors using the full covariance matrix via LL^T decomposition (Cholesky)

# Definining the lower triangular triangular matrix L and computing parameter perturbations
lowerTriangularMatrix = np.linalg.cholesky(covarianceMatrix)
firstPerturbedParameters = initialGuessCopy + lowerTriangularMatrix@np.random.randn(6)
secondPerturbedParameters = initialGuessCopy + lowerTriangularMatrix@np.random.randn(6)
thirdPerturbedParameters = initialGuessCopy + lowerTriangularMatrix@np.random.randn(6)

# Extracting the values from within the np.matrix object (problematic to work with)
firstParameters = np.array([firstPerturbedParameters.item(i) for i in range(6)])
secondParameters = np.array([secondPerturbedParameters.item(i) for i in range(6)])
thirdParameters = np.array([thirdPerturbedParameters.item(i) for i in range(6)])

# Computing the residuals of the best fit parameters for each perturbative realization
originalValues = tripleLorentzianFunction(initialGuessCopy,timeRange)[0]
firstValues = tripleLorentzianFunction(firstParameters,timeRange)[0]
secondValues = tripleLorentzianFunction(secondParameters,timeRange)[0]
thirdValues = tripleLorentzianFunction(thirdParameters,timeRange)[0]

# Plotting all three perturbed realizations and their respective resiuals
fig, axs = plt.subplots(2, 1, figsize=(10, 5), constrained_layout=True)

ax = axs[0]
ax.plot(timeRange,firstValues, color='darkviolet', label='First Realization')
ax.plot(timeRange,secondValues, '--',color='mediumspringgreen', label='Second Realization')
ax.plot(timeRange,thirdValues, ':', color='blue', label='Third Realization')
ax.set_ylabel('Signal')
ax.grid()
ax.legend()

ax = axs[1]
ax.plot(timeRange,originalValues-firstValues, color='darkviolet', label='First Realization')
ax.plot(timeRange,originalValues-secondValues, '--',color='mediumspringgreen', label='Second Realization')
ax.plot(timeRange,originalValues-thirdValues, ':', color='blue', label='Third Realization')
ax.set_ylabel('Residuals')
ax.set_xlabel('t')
ax.grid()
ax.legend()
plt.savefig('PlotIV.png',dpi=500)
plt.show()

# Computing the typical differennce in chisquare for perturbed parameters compared to the best fit chisquare
chisquareDifference = np.zeros((3))
parameterValues = [firstValues, secondValues, thirdValues]
for realization in range(3):
    bestFitChisquare =  np.sum((originalValues-signalValues)**2/σ**2)/(timeRange.shape[0]-6)
    realizationChisquare = np.sum((parameterValues[realization]-signalValues)**2/σ**2)/(timeRange.shape[0]-6)
    chisquareDifference[realization] = np.abs(bestFitChisquare-realizationChisquare)
print(np.mean(chisquareDifference))


# %% Computing the data best-fit using a MCMC algorithm (following Jon's class MCMC code)

# Defining the triple Lorentzian without the derivatives (simpler to work with)
def simpleTripleLorentzianFunction(p,t):
    
    # Defining parameters
    a, b, c, t0, omega, dt = p #Note: DIFFERENT ORDER
    
    # Defining the function itself
    functionValue = a/( 1 + ((t-t0)**2)/(omega**2) ) + b/( 1 + ((t-t0+dt)**2)/(omega**2) ) + c/( 1 + ((t-t0-dt)**2)/(omega**2) )
    
    return functionValue

# Defining a chisquare function to see whether or not the model is physical
def chiSquare(p,x,f,noiseValues):
    
    # Defining predicted value to compare to
    predictedValues = simpleTripleLorentzianFunction(p,timeRange) ###################################
    
    # Defining the outputted chisquare
    chisquareValue = np.sum(((f-predictedValues)/noiseValues)**2)
    
    return chisquareValue

# Defining the MCMC chain function
def MCMC(parameters,stepSize,t,f,function,noiseValues,numberOfSteps=1000):
    
    # Computing the initial chisquare of the model
    initialChisquare = function(parameters,t,f,noiseValues)
    
    # Initialization some arrays for the chain iteration chisquares
    chainArray = np.zeros([numberOfSteps,parameters.shape[0]])
    chisquareVector = np.zeros(numberOfSteps)
    
    # Iterating over each MCMC step
    for iteration in range(numberOfSteps):
        
        # Defining current (trial) parameters and the model's corresponding chisquare
        currentParameter = parameters + stepSize*np.random.randn(parameters.shape[0])
        currentChisquare = function(currentParameter,t,f,noiseValues)
        differentialChisquare = currentChisquare-initialChisquare
        
        # Defining acceptance probability conditions (Gaussian)
        acceptanceProbability = np.exp(-0.5*differentialChisquare)
        acceptanceCondition = np.random.rand(1)<acceptanceProbability
        
        # Defining whether or not the chain iteration is accepted
        if acceptanceCondition:
            
            # Updating the parameter and chisquare values
            parameters = currentParameter
            initialChisquare = currentChisquare
        chainArray[iteration,:] = parameters
        chisquareVector[iteration] = initialChisquare
    
    return chainArray, chisquareVector

# Computing the best-fit parameters via the MCMC algorithm 
previousBestFitParameters = initialGuessCopy
stepSizeChoice = np.array([9.14128147e-04, 8.70429036e-04, 8.52345103e-04, 1.08008106e-08, 1.93273356e-08, 1.30674361e-07])
outputChain, outputChisquareVector = MCMC(previousBestFitParameters,stepSizeChoice,timeRange,signalValues,chiSquare,σ,numberOfSteps=20000)

       
#%% Plotting the chain for the 'a' parameter
plt.plot(outputChain[:,0], color='darkviolet', label='a Parameter Chain')
plt.xlabel('Chain Index')
plt.ylabel('a')
plt.grid()
plt.legend()
plt.savefig('PlotV.png',dpi=500)


# %% Computing the new fitted parameters via computing the mean and standard deviations

# Computing the parameters and their uncertainties
aAverage = np.mean(outputChain[:,0])
aStandardDeviation = np.std(outputChain[:,0])
bAverage = np.mean(outputChain[:,1])
bStandardDeviation = np.std(outputChain[:,1])
cAverage = np.mean(outputChain[:,2])
cStandardDeviation = np.std(outputChain[:,2])
t0Average = np.mean(outputChain[:,3])
t0StandardDeviation = np.std(outputChain[:,3])
omegaAverage = np.mean(outputChain[:,4])
omegaStandardDeviation = np.std(outputChain[:,4])
dtAverage = np.mean(outputChain[:,5])
dtStandardDeviation = np.std(outputChain[:,5])

# Printing the best-fit parameters and their error
print('Best fit-parameter a: ', aAverage,aStandardDeviation)
print('Best fit-parameter b: ', bAverage,bStandardDeviation)
print('Best fit-parameter c: ', cAverage,cStandardDeviation)
print('Best fit-parameter t0: ', t0Average,t0StandardDeviation)
print('Best fit-parameter omega: ', omegaAverage,omegaStandardDeviation)
print('Best fit-parameter dt: ', dtAverage,dtStandardDeviation)


# %% Computing the width of the resonant cavity
cavityWidth = dtAverage/omegaAverage*9
widthUncertainty = cavityWidth * (1.8e-8/omegaAverage)


# %%
