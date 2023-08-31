#%% Importing modules & defining the null valued function calls
import numpy as np

#We will make these global variables within the argument of the functions to be able to use outside the function argument
integratorCalls = 0 
lazyCalls = 0


#%% Defining the 'lazy' integrator from the lecture
def lazyIntegrator(fun,a,b,tol):
    
    # Setting the call number to global so that it may be accessed outside the function argument
    global lazyCalls
    
    # Defining the range, measure, different area sizes
    x=np.linspace(a,b,5)
    y=fun(x)
    dx=(b-a)/(len(x)-1)
    area1=2*dx*(y[0]+4*y[2]+y[4])/3
    area2=dx*(y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])/3
    
    # Updating the amount of function calls per amount of points evaluated at
    lazyCalls += 5
    
    # Defining the error and putting a condition on the tolerance
    err=np.abs(area1-area2)
    if err<tol: 
        return area2
    else:
        c=(a+b)/2
        left=lazyIntegrator(fun,a,c,tol/2)
        right=lazyIntegrator(fun,c,b,tol/2)
        return left+right
    
    
#%% Defining the iterative custom integrator without recalling f
def integrate_adaptive(fun,a,b,tol,extra=None):
    
    # Setting the call number to global so that it may be accessed outside the function argument & defining the range
    global integratorCalls
    rangeValues = np.linspace(a,b,5)
    
    # Fixing the 'extra' parameter condition: only set to None for the first call
    if extra is None:
        yValue = fun(rangeValues)
        
        # Updating the amount of function calls per amount of points evaluated at
        integratorCalls += 5
        
    else:
        
        # Extracting intermediate points from the 'extra' parameter. Including fun(rangeValues[n]) for n=0,2 caused problematic results
        yValue = np.asarray([
            extra[0],fun(rangeValues[1]),extra[1],fun(rangeValues[3]),extra[2]
        ])
        
        # Updating the amount of function calls per amount of points evaluated at
        integratorCalls += 2
        
    # Defining the measure and the different area sizes
    differentialMeasure = (b-a)/4
    firstArea = 2*differentialMeasure*(yValue[0]+4*yValue[2]+yValue[4])/3
    secondArea = differentialMeasure*(yValue[0]+4*yValue[1]+2*yValue[2]+4*yValue[3]+yValue[4])/3  
    
    # Defining the error and putting a condition on the tolerance
    integrationError = np.abs(firstArea-secondArea)
    if integrationError < tol:
        return secondArea
    
    else:
        
        # Computing the next yValue via iteration: plugging in the previous yValue back into the integrator
        leftPiece=integrate_adaptive(fun,a,rangeValues[2],tol/2, np.asarray([yValue[0], yValue[1], yValue[2]]))
        rightPiece=integrate_adaptive(fun,rangeValues[2],b,tol/2, np.asarray([yValue[2], yValue[3], yValue[4]]))
        
        return leftPiece+rightPiece 
    
    
# %% Computing some examples for different functions to compare the function call numbers

# Defining the example functions
gaussianFunction = lambda x : np.exp((-x**2)/10)
sinusoidalFunction = lambda x : np.sin((x/10)-5)
rationalFunction = lambda x : x/((x**2)-1000)

# Computing the different amount of function calls (reseting the variable amounts after computation)
integrate_adaptive(gaussianFunction,0,10,1e-5)
lazyIntegrator(gaussianFunction,0,10,1e-5)
print("Lazy count: ", lazyCalls, "Integrator count: ", integratorCalls, "Difference: ", lazyCalls-integratorCalls)
lazyCalls = 0
integratorCalls = 0

integrate_adaptive(sinusoidalFunction,0,10,1e-5)
lazyIntegrator(sinusoidalFunction,0,10,1e-5)
print("Lazy count: ", lazyCalls, "Integrator count: ", integratorCalls, "Difference: ", lazyCalls-integratorCalls)
lazyCalls = 0
integratorCalls = 0

integrate_adaptive(rationalFunction,0,10,1e-5)
lazyIntegrator(rationalFunction,0,10,1e-5)
print("Lazy count: ", lazyCalls, "Integrator count: ", integratorCalls, "Difference: ", lazyCalls-integratorCalls)
lazyCalls = 0
integratorCalls = 0


# %%
