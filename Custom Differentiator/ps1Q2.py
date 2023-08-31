#%% Importing modules
import numpy as np
import matplotlib.pyplot as plt


# %% Defining the numerical differentiator
def ndiff(fun,x,full=False):
    
    # Defining required parameters
    ε = 10**(-15) #taking a small epsilon
    γ = 10**(-10) #taking a small gamma value
    
    # Defining different derivatives and the error function
    firstDerivative = lambda k,h : ( fun(x+h)-fun(x-h) )/(2*h) #the variable k is used in place for x to avoid issues
    thirdDerivative = lambda k,h : ( fun(x+2*h) - fun(x-2*h) -2*fun(x+h) + 2*fun(x-h) )/(2*h**3)
    errorFunction = lambda k,h : ((h**2)/6)*thirdDerivative(k,h) + (ε*fun(k))/h
    
    # Defining optimal h function
    optimalHFunction = lambda k : np.cbrt( (3*ε*fun(x))/(thirdDerivative(x,γ)) )
    
    # Checking whether or not full is True/False
    if full == False:
        
        # Computing the optimal h value at x
        optimalH = optimalHFunction(x)
        
        return firstDerivative(x,optimalH)
        
    if full == True:
        
        # Computing the optimal h value at x and the error
        optimalH = optimalHFunction(x)
        error = errorFunction(x,optimalH)
        
        return (firstDerivative(x,optimalH),error)
    
        
#%% Testing the above differentiator

# Defining the test function (sinx) and its derivative (cosx)
testFunction = np.sin
derivativeFunction = np.cos

# Defining a linspace over which we evaluate the derivative with our function and the analytic function
testPoints = np.linspace(0,1,100)
differentiatorPoints = ndiff(testFunction,testPoints)
realPoints = derivativeFunction(testPoints)
errorInPoints = ndiff(testFunction,testPoints,True)[1]

# Plotting the difference between the differentiator and actual derivative points
fig, axs = plt.subplots(2, 1, figsize=(10, 5), constrained_layout=True)

ax = axs[0]
ax.scatter(testPoints,differentiatorPoints-realPoints, color='darkviolet', label='Difference Points')
ax.set_ylabel('Function Difference')
ax.grid()
ax.legend()

ax = axs[1]
ax.scatter(testPoints,errorInPoints, color='mediumspringgreen', label='Error')
ax.set_ylabel('Error')
ax.set_xlabel('x')
ax.grid()
ax.legend()
plt.savefig('functionDifference.png',dpi=500)
plt.show()


# %%
