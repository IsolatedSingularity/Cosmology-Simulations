#%% Importing modules
import numpy as np
import matplotlib.pyplot as plt


#%% Defining parameters and the error function

# Defining paramteres
epsilon = 10**(-15)
criticalDelta = (2025/198)**(1/10)*(epsilon**(1/5))
secondCriticalDelta = (2025/198)**(1/10)*((epsilon**(1/5)/100))

# Defining the error functions for f(x)=exp(x),exp(x/100)
errorFunction = lambda x : (49/8100) * np.exp(2*x) * (criticalDelta**8) + epsilon**2 * (np.exp(2*x)/(criticalDelta**2))
secondErrorFunction = lambda x : (49/8100) * ((1/100)**5 * np.exp(x/100))**2 * (secondCriticalDelta**8) + epsilon**2 * (np.exp(x/50)/(secondCriticalDelta**2))


#%% Plotting & saving the error plots
xValues = np.linspace(0,1,100)
plt.plot(xValues, errorFunction(xValues), label = r'$f(x) = e^{x}$', color='mediumspringgreen')
plt.plot(xValues, secondErrorFunction(xValues), label = r'$f(x) = e^{x/100}$', color='darkviolet')
plt.xlabel('x')
plt.ylabel('Error')
plt.grid()
plt.legend()
plt.savefig('errorFunctions.png',dpi=500)
plt.show()


#%% Repeating the above for a logarithmic plane
xValues = np.linspace(0,1,100)
plt.loglog(xValues, errorFunction(xValues), label = r'$f(x) = e^{x}$', color='mediumspringgreen')
plt.loglog(xValues, secondErrorFunction(xValues), label = r'$f(x) = e^{x/100}$', color='darkviolet')
plt.xlabel('x')
plt.ylabel('Error')
plt.grid()
plt.legend()
plt.savefig('loglogErrorFunctions.png',dpi=500)
plt.show()


#%% Plotting the error function with fixed x=0 as a function of delta

# Defining the error functions for a fixed x=0 and free δ parameter
errorFunctionDelta = lambda δ : (49/8100) * np.exp(2*0) * (δ**8) + epsilon**2 * (np.exp(2*0)/(δ**2))
secondErrorFunctionDelta = lambda δ : (49/8100) * ((1/100)**5 * np.exp(0/100))**2 * (δ**8) + epsilon**2 * (np.exp(0/50)/(δ**2))

# Plotting the error function for fixed x=0
δValues = np.linspace(0.0001,0.0013,100)
plt.plot(δValues, errorFunctionDelta(δValues), label = r'$f(x) = e^{x}$', color='mediumspringgreen')
plt.plot(δValues, secondErrorFunctionDelta(δValues), label = r'$f(x) = e^{x/100}$', color='darkviolet')
plt.scatter(criticalDelta,errorFunction(criticalDelta), label=r'$\delta_0$', color='blue')
plt.scatter(secondCriticalDelta,errorFunction(secondCriticalDelta), label=r"$\delta'_0$", color='black')
plt.xlabel('δ')
plt.ylabel('Error')
plt.grid()
plt.legend()
plt.savefig('deltaErrorFunctions.png',dpi=500)
plt.show()


# %%
