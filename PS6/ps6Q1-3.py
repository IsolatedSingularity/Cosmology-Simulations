#%% Importing modules
import numpy as np
import matplotlib.pyplot as plt


# %% Defining a function that shifts a arbitrary array by a set amount using a convolution
def arrayShifter(array, shift):
    
    # Defining the length of the array to be shifted
    arrayLength = len(array)
    
    # Imposing a modulo condition on the shift if it is longer than the array
    if shift > arrayLength:
        shift %= arrayLength
        
    # Defining the shift array 'g'
    shiftArray = np.zeros(arrayLength)
    shiftArray[shift] = 1
    
    # Performing the shift via the Fourier representation of the convolution
    convolutionShift = np.fft.irfft(
        np.fft.fft(array)*np.fft.fft(shiftArray),arrayLength
    )
    
    return convolutionShift


# %% Defining the Gaussian, shifting it, and plotting the array shift

# Defining the range of the function
xRange = np.linspace(-5,5,1001)

# Defining the Gaussian function
Gaussian = lambda x : np.exp(-0.5*x**2)
gaussianArray = Gaussian(xRange)

# Defining the shifted Gaussian
shiftedGaussian = arrayShifter(gaussianArray, int(len(xRange)/2))

# Plotting the results of the shift
plt.plot(xRange,gaussianArray, color='mediumspringgreen', label='Gaussian')
plt.plot(xRange,shiftedGaussian, color='darkviolet', label='Shifted Gaussian')
plt.xlabel('x')
plt.ylabel('Function Value')
plt.grid()
plt.legend()
plt.savefig('PlotI.png',dpi=500)


# %% Defining the cross correlation function routine
def arrayCorrelation(firstArray, secondArray):
    
    # Defining the correlation of the arrays using the Fourier representation
    rawCorrelation = np.fft.irfft(
        np.fft.fft(firstArray)*np.conj(np.fft.fft(secondArray)), len(firstArray)
    )
    
    # Defining the normalized cross correlation
    crossCorrelation = rawCorrelation / np.max(rawCorrelation)
    
    return crossCorrelation

# Defining the correlation of the Gaussian with itself
autocorrelatedGaussian = arrayCorrelation(gaussianArray,gaussianArray)

# Plotting the Gaussian and the autocorrelated Gaussian
plt.plot(xRange,gaussianArray, color='mediumspringgreen', label='Gaussian')
plt.plot(xRange,autocorrelatedGaussian, color='darkviolet', label='Gaussian Autocorrelation')
plt.xlabel('x')
plt.ylabel('Function Value')
plt.grid()
plt.legend()
plt.savefig('PlotII.png',dpi=500)


# %% Computing the autocorrelation of arbitrarily shifted Gaussians

# Defining the the shifted autocorrelation function
def shiftedCorrelation(array, shift):
    
    # Performing the correlation after taking a shift
    shiftCorrelationArray = arrayCorrelation(array, arrayShifter(array,shift))
    
    return shiftCorrelationArray

# Shifting the Gaussian by arbitrary amounts and plotting the correlation
arbitraryShifts = [100,300,500,700]
plt.plot(xRange, autocorrelatedGaussian, label='Shift by 0')
for shift in arbitraryShifts:
    
    # Labelling the different plots for each shift
    plotLabel = 'Shifted by ' + str(shift)
    plt.plot(xRange, shiftedCorrelation(gaussianArray,shift), label=plotLabel)
    
plt.xlabel('x')
plt.ylabel('Function Value')
plt.grid()
plt.legend()
plt.savefig('PlotIII.png',dpi=500)
 

# %% Defining and plotting the safe convolution routine

# Defining the safe convolution function for arrays of two varying sizes
def safeConvolution(firstArray, secondArray):
    
    # Defining the size of the arrays
    firstSize = len(firstArray)
    secondSize = len(secondArray)
    sizeDifference = np.abs(firstSize-secondSize)
    
    # Defining the zero array to pad the difference
    paddingArray = np.zeros(sizeDifference)
    
    # Determining which array is larger and increasing the size of the smaller one
    if firstSize > secondSize:
        secondArray = np.append(secondArray, paddingArray)
    if secondSize > firstSize:
        firstArray = np.append(firstArray, paddingArray)
        
    # Further padding both arrays to prevent overlap
    paddedFirstArray = np.append(firstArray, paddingArray)
    paddedSecondArray = np.append(secondArray, paddingArray)
    
    # Performing the Fourier representation of the convolution
    convolution = np.fft.irfft(
        np.fft.rfft(paddedFirstArray)*np.fft.rfft(paddedSecondArray)
    )
        
    return convolution

# Defining the safe and unsafe convolutions of different length Gaussians
firstGaussian = Gaussian(
    np.linspace(-5,5,1001)
)
secondGaussian = Gaussian(
    np.linspace(-5,5,2001)
)
safeConvolutionArray = safeConvolution(firstGaussian,secondGaussian)

# Plotting the different convolutions
plt.plot(
    np.linspace(-5,5,3000),safeConvolutionArray, label='Safe Convolved Gaussian', color='darkviolet'
    )
plt.xlabel('x')
plt.ylabel('Function Value')
plt.grid()
plt.legend()
plt.savefig('PlotIV.png',dpi=500)


# %%
