#%% Importing modules
import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import chebfit, chebval


# %% Defining the base-2 logarithmic fit with Chebyshev polynomials & plotting

# Defining the base and translated ranges
xRange = np.linspace(1/2,1,1000)
translatedRange = 4*xRange - 3

# Defining the logarithm function and fitting it
baseTwoLogarithm = np.log2(xRange)
polynomialChebyshev = chebfit(translatedRange,baseTwoLogarithm,7)
computedPointChebyshev = chebval(translatedRange,polynomialChebyshev)

# Plotting the function, the fit, and their difference (accuracy/error)
fig, axs = plt.subplots(2, 1, figsize=(10, 5), constrained_layout=True)

ax = axs[0]
ax.plot(translatedRange,baseTwoLogarithm, color='mediumspringgreen', label='Logarithm')
ax.plot(translatedRange,computedPointChebyshev, '--', color='darkviolet', label='Chebyshev Fit')
ax.set_ylabel('Function Value')
ax.grid()
ax.legend()

ax = axs[1]
ax.plot(translatedRange,np.abs(baseTwoLogarithm-computedPointChebyshev), color='blue', label='Error')
ax.set_ylabel('Error')
ax.set_xlabel('x')
ax.grid()
ax.legend()
plt.savefig('PlotII.png',dpi=500)
plt.show()


#%% Defining the natrual logarithmic function for any x
def mylog2(x):
    
    # Splitting x into its mantissa and exponent: x=mantissa*2^exponent, and e for basis change
    mantissa, exponent = np.frexp(x)
    eMantissa, eExponent = np.frexp(math.e)
    
    # Defining the output of log_2x
    baseTwoNumber = chebval(4*mantissa-3,polynomialChebyshev) + exponent # log_2m + n
    baseTwoE = chebval(4*eMantissa-3,polynomialChebyshev) + eExponent
    
    # Converting to the natural logarithm basis
    naturalNumber = baseTwoNumber/baseTwoE
    
    return naturalNumber
    
    
#%% Plotting mylog2, numpy's natural logarithm, and their error difference
fig, axs = plt.subplots(2, 1, figsize=(10, 5), constrained_layout=True)
newRange = np.linspace(1,100,1000)

ax = axs[0]
ax.plot(newRange,np.log(newRange), color='mediumspringgreen', label='Numpy Logarithm')
ax.plot(newRange,mylog2(newRange), '--', color='darkviolet', label='mylog2')
ax.set_ylabel('Function Value')
ax.grid()
ax.legend()

ax = axs[1]
ax.plot(newRange,np.abs(np.log(newRange)-mylog2(newRange)), color='blue', label='Error')
ax.set_ylabel('Error')
ax.set_xlabel('x')
ax.grid()
ax.legend()
plt.savefig('PlotIII.png',dpi=500)
plt.show()


# %% Doing all these computations again but with Legendre polynomials

# Importing the Legendre module
from numpy.polynomial import legendre

# Defining the coefficients and evaluated points
legendreCoefficients = legendre.legfit(translatedRange,baseTwoLogarithm,7)

# Defining the logarithmic fit with Legendre polynomials
def legendreLogarithm(x):
    
    # Splitting x into its mantissa and exponent: x=mantissa*2^exponent, and e for basis change
    mantissa, exponent = np.frexp(x)
    eMantissa, eExponent = np.frexp(math.e)
    
    # Defining the output of log_2x
    baseTwoNumber = legendre.legval(4*mantissa-3,polynomialChebyshev) + exponent # log_2m + n
    baseTwoE = legendre.legval(4*eMantissa-3,polynomialChebyshev) + eExponent
    
    # Converting to the natural logarithm basis
    naturalNumber = baseTwoNumber/baseTwoE
    
    return naturalNumber

# Defining the numpy and custom logarithm values 
numpyValues = np.log(newRange)
legendreLogarithmValues = legendreLogarithm(newRange)
chebyshevLogarithmValues = mylog2(newRange)

# Defining the RMS function and the RMS values
rmsFunction = lambda fittedValues,realValues : np.sqrt(1/(fittedValues.shape[0])
                                                       *(realValues-fittedValues)**2)
chebyshevRMS = rmsFunction(chebyshevLogarithmValues,numpyValues)
legendreRMS = rmsFunction(legendreLogarithmValues,numpyValues)

# Plotting the function fits, and their RMS error
fig, axs = plt.subplots(3, 1, figsize=(10, 5), constrained_layout=True)

ax = axs[0]
ax.plot(newRange,numpyValues, color='mediumspringgreen', label='Numpy Logarithm')
ax.plot(newRange,legendreLogarithmValues, '--', color='darkviolet', label='Legendre Fit')
ax.plot(newRange,chebyshevLogarithmValues, ':', color='blue', label='Chebyshev Fit')
ax.set_ylabel('Function Value')
ax.grid()
ax.legend()

ax = axs[1]
ax.plot(newRange,chebyshevRMS, color='blue', label='Chebyshev Error')
ax.set_ylabel('RMS Error')
ax.grid()
ax.legend()

ax = axs[2]
ax.plot(newRange,legendreRMS, color='darkviolet', label='Legendre Error')
ax.set_ylabel('RMS Error')
ax.set_xlabel('x')
ax.grid()
ax.legend()
plt.savefig('PlotIV.png',dpi=500)
plt.show()

# Printing the max RMS of the two
print('Max Legendre RMS: ', np.max(legendreRMS))
print('Max Chebyshev RMS: ', np.max(chebyshevRMS))


# %%
