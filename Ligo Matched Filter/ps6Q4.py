#%% Importing modules
import numpy as np
import matplotlib.pyplot as plt


# %% Defining our 'analytic' estimate of the discrete Fourier transform

# Fixing N, k, and the x range
N = 100
k = 8*np.pi #bar K
xRange = np.arange(N)

# Defining the analytic DFT estimate
def analyticDFT(array):
    
    # Defining the array size, the range of x, and initializing the DFT
    arraySize = len(array)
    xRange = np.arange(arraySize)
    analyticDFT = np.zeros(
        arraySize, dtype = np.complex128
    )
    
    # Performing the sum over k (bar k) and x with two loops
    for k in range(arraySize):
        for x in range(arraySize):
            analyticDFT[k] += array[x] * np.exp(
                -2j*np.pi*k*xRange[x]/arraySize
            )
    
    return analyticDFT


# %% Defining the function, the numerical estimate, and plotting the two

# Defining the function to be transformed and the range of k (bar K)
sineFunction = np.sin(2*np.pi*k*xRange/N)
kRange = np.arange(len(sineFunction))[:N//2+1]

# Defining the numerical and analytic estimates
numericalEstimate = np.fft.rfft(sineFunction)
analyticEstimate = analyticDFT(sineFunction)

# Plotting the two function estimates and their difference
fig, axs = plt.subplots(2, 1, figsize=(10, 5), constrained_layout=True)

ax = axs[0]
ax.plot(kRange,np.abs(analyticEstimate[:N//2+1]), label = 'Analytic result', color='mediumspringgreen')
ax.plot(kRange, np.abs(numericalEstimate), '--', label = 'Numerical result', color='darkviolet')
ax.set_ylabel('Discrete Fourier Transform')
ax.grid()
ax.legend()

ax = axs[1]
ax.plot(kRange, np.abs(np.abs(analyticEstimate[:N//2+1])-np.abs(numericalEstimate)), color='blue', label='Analytic - Numerical')
ax.set_ylabel('Difference')
ax.set_xlabel(r'$\bar{k}$')
ax.grid()
ax.legend()
plt.savefig('PlotV.png',dpi=500)


# %% Defining and plotting the results of a window function

# Defining the window function to cut the leaking modes
windowFunction = 1/2 - 1/2 * np.cos(2*np.pi*xRange/N)

# Multiplying the function by the window function and computing the DFT
windowedSineFunction = sineFunction * windowFunction
windowedAnalyticEstimate = analyticDFT(windowedSineFunction)
windowedNumericalEstimate = np.fft.rfft(windowedSineFunction)

# Plotting the result of the window
fig, axs = plt.subplots(2, 1, figsize=(10, 5), constrained_layout=True)

ax = axs[0]
ax.plot(kRange,np.abs(windowedAnalyticEstimate[:N//2+1]), label = 'Analytic result', color='mediumspringgreen')
ax.plot(kRange, np.abs(windowedNumericalEstimate), '--', label = 'Numerical result', color='darkviolet')
ax.set_ylabel('Discrete Fourier Transform')
ax.grid()
ax.legend()

ax = axs[1]
ax.plot(kRange, np.abs(np.abs(windowedAnalyticEstimate[:N//2+1])-np.abs(windowedNumericalEstimate)), color='blue', label='Analytic - Numerical')
ax.set_ylabel('Difference')
ax.set_xlabel(r'$\bar{k}$')
ax.grid()
ax.legend()
plt.savefig('PlotVI.png',dpi=500)


# %%
