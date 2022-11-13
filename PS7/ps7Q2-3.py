#%% Importing modules
import numpy as np
import matplotlib.pyplot as plt


# %% Defining and plotting the different potential exponential deviates

# Definining the different exponential deviates
gaussian = lambda x,σ : np.exp(-(x)**2/(2*σ**2))
lorentzian = lambda x : 1/(1 + x**2)
powerLaw = lambda x,a,b,c : (x+b)**(-a) + c
exponential = lambda x : np.exp(-x)

# Plotting the different deviate functions for arbitrarily fixed parameters
xRange = np.linspace(0,10,100)
plt.plot(xRange, exponential(xRange), color = 'mediumspringgreen', label= 'Exponential')
plt.plot(xRange, lorentzian(xRange), color = 'darkviolet', label= 'Lorentzian')
plt.plot(xRange, gaussian(xRange, 3/2), color = 'blue', label= 'Gaussian')
plt.plot(xRange, powerLaw(xRange, 0.7, 0.75, -0.2), color = 'black', label= 'Power Law')
plt.xlabel('x')
plt.ylabel('Deviate Value')
plt.grid()
plt.legend()
plt.savefig('PlotIII.png',dpi=500)


# %% Sampling many points, restricting our selection using the rejection condition

# Sampling many points and computing the function values at those points
randomSampling = np.random.rand(100000)
lorentzValues = lorentzian(randomSampling)
expontialValues = exponential(randomSampling)

# Computing the ratio and restricting the selection based on the rejection condition
functionRatio = expontialValues/lorentzValues
rejectionCondition = functionRatio < randomSampling
acceptedPoints = functionRatio[rejectionCondition]

# Plotting the histogram
plt.hist(acceptedPoints, bins=50, color='darkviolet', label='Histogram')
plt.grid()
plt.legend()
plt.plot()
plt.savefig('PlotVI.png',dpi=500)


# %% Repeating the above but for a ratio-of-uniforms generator

# Defining N, u, and v
N = int(1e5)
u = np.random.rand(N)
v = np.random.rand(N)

# Computing an upper bound on u
upperBoundU = np.sqrt(np.exp(-(v/u)))

# Keeping points only following the condition: 0 <= u <= upperBoundU
keptPoints = u < upperBoundU

# Updating the amount of kept points
u = u[keptPoints]
v = v[keptPoints]

# Plotting the results of the kept points
plt.hist(v/u, bins=50, color='darkviolet', label='Histogram')
plt.grid()
plt.legend()
plt.plot()
plt.savefig('PlotV.png',dpi=500)

# Printing the fraction of kept numbers
print('Amount of kept numbers: ', len(u)/len(upperBoundU))


# %%
