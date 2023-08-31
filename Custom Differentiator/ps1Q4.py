#%% Importing modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splev, splrep
from scipy.optimize import curve_fit


# %% Defining the function and the rational function fits

# Defining the functions and corresponding ranges
cosineFunction = np.cos
lorentzianFunction = lambda x : 1/(1+x**2)
angleRange = np.linspace(-np.pi/2,np.pi/2,100)
lorentzRange = np.linspace(-1,1,100)
cosineValues = cosineFunction(angleRange)
lorentzValues = lorentzianFunction(lorentzRange)

# Defining a generic rational and third order rational function (once again)
def rationalFunction(x, numeratorPolynomial, denominatorPolynomial):
    return np.polyval(numeratorPolynomial, x) / np.polyval(denominatorPolynomial + [1.0], x)
def thirdOrderRationalFunction(x, p0, p1, p2, q1, q2):
    return rationalFunction(x, [p0, p1, p2], [q1, q2])


#%% Performing the various fits of the two functions --

# Performing spline fits
splineCosineRepresentation = splrep(angleRange,cosineValues)
splineCosineFit = splev(angleRange,splineCosineRepresentation)
splineLorentzRepresentation = splrep(lorentzRange,lorentzValues)
splineLorentzFit = splev(lorentzRange,splineLorentzRepresentation)

# Performing polynomial fits
polynomialCosineObject = np.polyfit(angleRange,cosineValues,5)
polynomialCosineFit = np.poly1d(polynomialCosineObject)
polynomialLorentzObject = np.polyfit(lorentzRange,lorentzValues,5)
polynomialLorentzFit = np.poly1d(polynomialLorentzObject)

# Performing rational fits
poptCosine, pcovCosine = curve_fit(thirdOrderRationalFunction,angleRange,cosineValues)
poptLorentz, pcovLorentz = curve_fit(thirdOrderRationalFunction,lorentzRange,lorentzValues)


# %% Plotting the fits of the functions + their differences: cosine function
fig, axs = plt.subplots(2, 1, figsize=(10, 5), constrained_layout=True)

ax = axs[0]
ax.scatter(angleRange,cosineValues,color='darkviolet',label='True Points')
ax.plot(angleRange,splineCosineFit, label='Spline Fit', color='mediumspringgreen')
ax.plot(angleRange,polynomialCosineFit(angleRange),color='red',label='Polynomial Fit')
ax.plot(angleRange,thirdOrderRationalFunction(angleRange, *poptCosine), label='Rational Fit', color='blue')
ax.set_ylabel('Cos(x)')
ax.grid()
ax.legend()

ax = axs[1]
ax.plot(angleRange,cosineValues-splineCosineFit, label='Spline Fit', color='mediumspringgreen')
ax.plot(angleRange,cosineValues-polynomialCosineFit(angleRange),color='red',label='Polynomial Fit')
ax.plot(angleRange,cosineValues-thirdOrderRationalFunction(angleRange, *poptCosine), label='Rational Fit', color='blue')
ax.set_ylabel('Relative Difference')
ax.set_xlabel('x')
ax.grid()
ax.legend()
plt.savefig('cosineDifference.png',dpi=500)
plt.show()


# %%
# %% Plotting the fits of the functions + their differences: cosine function
fig, axs = plt.subplots(2, 1, figsize=(10, 5), constrained_layout=True)

ax = axs[0]
ax.scatter(lorentzRange,lorentzValues,color='darkviolet',label='True Points')
ax.plot(lorentzRange,splineLorentzFit, label='Spline Fit', color='mediumspringgreen')
ax.plot(lorentzRange,polynomialLorentzFit(lorentzRange),color='red',label='Polynomial Fit')
ax.plot(lorentzRange,thirdOrderRationalFunction(lorentzRange, *poptLorentz), label='Rational Fit', color='blue')
ax.set_ylabel('Lorentzian(x)')
ax.grid()
ax.legend()

ax = axs[1]
ax.plot(lorentzRange,lorentzValues-splineLorentzFit, label='Spline Fit', color='mediumspringgreen')
ax.plot(lorentzRange,lorentzValues-polynomialLorentzFit(lorentzRange),color='red',label='Polynomial Fit')
ax.plot(lorentzRange,lorentzValues-thirdOrderRationalFunction(lorentzRange, *poptLorentz), label='Rational Fit', color='blue')
ax.set_ylabel('Relative Difference')
ax.set_xlabel('x')
ax.grid()
ax.legend()
plt.savefig('lorentzDifference.png',dpi=500)
plt.show()


# %%
