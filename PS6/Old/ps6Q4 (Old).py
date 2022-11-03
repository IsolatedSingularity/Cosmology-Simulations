#%% Importing modules
import numpy as np
import matplotlib.pyplot as plt


# %% Defining the fixed value of k and N
k = np.pi
N = 100


# %% Defining the analytic result function
def analyticResult(barK):
    
    # Defining the first and second exponential terms
    firstExponential = (1/2j)*(1-np.exp(2j*np.pi*(k*N-barK)))/(1-np.exp(2j*np.pi*(k-barK/N)))
    secondExponential = (1/2j)*(1-np.exp(2j*np.pi*(-k*N-barK)))/(1-np.exp(2j*np.pi*(-k-barK/N)))
    
    # Combining the two
    combinedRealExponents = firstExponential+secondExponential
    
    return combinedRealExponents


# %% Defining the numerical result and plotting the two alongside their difference

# Defining the range of x & barK, and the sine function
xRange = np.arange(N)
sineFunction = np.sin(2*np.pi*k*xRange)
kRange = np.arange(len(sineFunction))[:N//2+1]

# Defining the analytic result, the FFT result and plotting the two
analyticDFT = analyticResult(kRange)
numericalDFT = np.fft.rfft(sineFunction)

fig, axs = plt.subplots(2, 1, figsize=(10, 5), constrained_layout=True)

ax = axs[0]
ax.plot(kRange,np.abs(analyticDFT), label = 'Analytic result', color='mediumspringgreen')
ax.plot(kRange, np.abs(numericalDFT), '--', label = 'Numerical result', color='darkviolet')
ax.set_ylabel('Discrete Fourier Transform')
ax.grid()
ax.legend()

ax = axs[1]
ax.plot(kRange, np.abs(np.abs(analyticDFT)-np.abs(numericalDFT)), color='blue', label='Analytic - Numerical')
ax.set_ylabel('Difference')
ax.set_xlabel(r'$\bar{k}$')
ax.grid()
ax.legend()
plt.savefig('PlotV.png',dpi=500)


# %%

# %%
