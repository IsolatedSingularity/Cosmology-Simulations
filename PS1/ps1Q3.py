#%% Importing modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import splev, splrep
from scipy.optimize import curve_fit


#%% Defining the routine function
dat = np.loadtxt('./lakeshore.txt')

def lakeshore(V,data):
    
    # Extraction of voltage and temperatures from the data
    voltageValues = data.T[1]
    temperatureValues = data.T[0]
    
    # Performing a cubic spline on the data
    splineFit = interp1d(voltageValues,temperatureValues, kind='cubic')

    # Checking if the input V is a number or an array
    if np.isscalar(V) == True:
        return splineFit(V)
    
    else: 
        voltageValues = []
        
        # Extracting all wanted temperature values pointwise
        for voltage in range(len(V)):
            voltageValues.append(splineFit(V[voltage]))
            
        return voltageValues
    
    
# %% Error analysis: derivative error

# Opening the text file once more
textFile = np.loadtxt('./lakeshore.txt')

# Extracting temperature, voltage, and dV/dT values
voltageValues = textFile.T[1]
temperatureValues = textFile.T[0]
derivativeValues = textFile.T[2]/1000 #The /1000 is to match the units of the derivative spline

# Fitting an antisymmetric cubic spline through the inverted data
splineRepresentation = splrep(temperatureValues, voltageValues)
splineFit = splev(temperatureValues, splineRepresentation)

# Plotting data to see how it looks
plt.scatter(temperatureValues, voltageValues, color='darkviolet',label='Data points')
plt.plot(temperatureValues,splineFit,color='mediumspringgreen',label='Spline fit')
plt.xlabel('Temperature')
plt.ylabel('Voltage')
plt.legend()
plt.grid()
plt.savefig('voltageTemperature.png',dpi=500)
plt.show()

# Computing the derivatives at each point of the spline, creating a new spline for dV/dT
derivativeOfSpline = splev(temperatureValues,splineRepresentation,der=1)

# Plotting the actual derivative values, the calculated spline derivative values, and the difference (error)
fig, axs = plt.subplots(2, 1, figsize=(10, 5), constrained_layout=True)

ax = axs[0]
ax.plot(temperatureValues,derivativeValues, color='darkviolet', label='Actual Derivatives')
ax.plot(temperatureValues,derivativeOfSpline, color='mediumspringgreen', label='Spline Derivatives')
ax.set_ylabel('dV/dT')
ax.grid()
ax.legend()

ax = axs[1]
ax.plot(temperatureValues,derivativeValues-derivativeOfSpline, color='blue', label='Difference')
ax.set_ylabel('Difference')
ax.set_xlabel('Temperature')
ax.grid()
ax.legend()
plt.savefig('deriavtiveDifference.png',dpi=500)
plt.show()


# %% Error analysis: spline fit error

# Computing a polynomial fit of the data ---
polynomialObject = np.polyfit(temperatureValues, voltageValues, 20)
polynomialFit = np.poly1d(polynomialObject)

# Computing a rational function fit of the data ---

#Defining a generic & third order rational function
def rationalFunction(x, numeratorPolynomial, denominatorPolynomial):
    return np.polyval(numeratorPolynomial, x) / np.polyval(denominatorPolynomial + [1.0], x)
def thirdOrderRationalFunction(x, p0, p1, p2, q1, q2):
    return rationalFunction(x, [p0, p1, p2], [q1, q2])

# Performing best fit of the rational function to the data
popt, pcov = curve_fit(thirdOrderRationalFunction, temperatureValues, voltageValues)

# Defining the average of the three fits; the 'true' value
averageValues = (splineFit+polynomialFit(temperatureValues)+thirdOrderRationalFunction(temperatureValues, *popt))/3

# Plotting the spline, polynomial, and rational function fits to the data to viasualize:
plt.scatter(temperatureValues, voltageValues, label='Original Points', color='darkviolet')
plt.plot(temperatureValues, thirdOrderRationalFunction(temperatureValues, *popt), label='Rational Fit')
plt.plot(temperatureValues,polynomialFit(temperatureValues),color='red',label='Polynomial Fit')
plt.plot(temperatureValues,splineFit,color='mediumspringgreen',label='Spline Fit')
plt.plot(temperatureValues,averageValues, label='Average Fit', color='black')
plt.grid()
plt.legend()
plt.savefig('allFits.png',dpi=500)


# %% Plotting the difference between the average fit and the spline fit
fig, axs = plt.subplots(2, 1, figsize=(10, 5), constrained_layout=True)

ax = axs[0]
ax.plot(temperatureValues,splineFit,color='mediumspringgreen',label='Spline Fit')
ax.plot(temperatureValues,averageValues, label='Average Fit', color='black')
ax.set_ylabel('Voltage')
ax.grid()
ax.legend()

ax = axs[1]
ax.plot(temperatureValues,splineFit-averageValues, color='blue', label='Difference')
ax.set_ylabel('Difference')
ax.set_xlabel('Temperature')
ax.grid()
ax.legend()
plt.savefig('splineDifference.png',dpi=500)
plt.show()


# %%
