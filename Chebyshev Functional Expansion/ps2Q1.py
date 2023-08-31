#%% Importing modules
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate


# %% Defining a custom integrator over theta for a fixed z
def fixedZIntegrator(function,a,b,intervals,z):
    
    # Defining the constant measure step size and the range
    differentialMeasure = (b-a)/intervals
    range = np.linspace(a,b,intervals)
    
    # Computing the integral using Simpson's rule while looping over different values of z
    integralValue = differentialMeasure * (function(range,z) + 4*function(range+differentialMeasure/2,z) +
                                           function(range+differentialMeasure,z))/6
    
    return np.sum(integralValue)


#%% Defining the radius of the shell, the differential electric field, and the analytic electric field, respectively

R = 100

electricDifferential = lambda theta,z : ((R**2)/2)*(np.sin(theta)*(z-R*np.cos(theta)))/(
    (R**2+z**2-2*R*z*np.cos(theta))**(3/2))

def analyticElectricField(z):
    result = ((R**2)/(2*z**2)) * ((z-R)/(np.abs(z-R))+(z+R)/(np.abs(z+R))) if z != R else 1
    return result #Notice I set E(z=R)=1 to avoid the division by zero error


# %% Computing the integral with the integrator and scipy & plotting the results and the error

# Defining the range of z over which dE is integrated over
zRange = np.linspace(1,200,199)

# Computations of the integral via the custom integrator and scipy, respectively
integratorElectricField = lambda z : fixedZIntegrator(electricDifferential,0,np.pi,10000,z)
scipyElectricField = lambda z : integrate.quad(electricDifferential,0,np.pi,args=(z,))[0]

# Defining the values of the theta integral for different z
analyticValues = np.array([analyticElectricField(z) for z in range(1,len(zRange)+1)])
integratorValues = np.array([integratorElectricField(z) for z in range(1,len(zRange)+1)])
scipyValues = np.array([scipyElectricField(z) for z in range(1,len(zRange)+1)])

# Plotting the analytic electric field, the integral results, and the error of the analytic results
fig, axs = plt.subplots(2, 1, figsize=(10, 5), constrained_layout=True)

ax = axs[0]
ax.plot(zRange, analyticValues, color='darkviolet', label='Analytic Result')
ax.plot(zRange, integratorValues, color='mediumspringgreen', label='Integrator Result')
ax.plot(zRange, scipyValues[0:], '--',color='blue', label='Scipy Result')
ax.set_ylabel('Electric Field')
ax.grid()
ax.legend()

ax = axs[1]
ax.plot(zRange,analyticValues-integratorValues, color='mediumspringgreen', label='Integrator Error')
ax.plot(zRange,analyticValues-scipyValues, '--',color='blue', label='Scipy Error')
ax.set_ylabel('Error')
ax.set_xlabel('z')
ax.grid()
ax.legend()
plt.savefig('PlotI.png',dpi=500)
plt.show()


# %%
