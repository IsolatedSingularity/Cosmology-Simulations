#%% Importing modules
import numpy as np
import matplotlib.pyplot as plt

# %% Defining the right hand side of the ODE and the analytic solution
rightHandSide = lambda x,y : y/(1+x**2)
analyticSolution = lambda x : (1/np.exp(np.arctan(-20))) * np.exp(np.arctan(x))

# %% Defining the fourth order Runge-Kutta step, integrating it, and plotting the difference

# Defining the initial RG step
def rk4_step(fun,x,y,h):
    
    # Defining the kn parameters for n=[1,4]
    k1 = fun(x,y) * h
    k2 = fun(x+h/2,y+k1/2) * h
    k3 = fun(x+h/2, y+k2/2) * h
    k4 = fun(x+h, y+k3) * h
    
    return y+(k1+2*k2+2*k3+k4)/6

# Performing the integration using the step over the asked range
integrationRange = np.linspace(-20,20,200)
integratedSolution = np.zeros(len(integrationRange))
integratedSolution[0] = 1
stepSize = integrationRange[1] - integrationRange[0]
for points in range(len(integrationRange)-1):
    integratedSolution[points+1] = rk4_step(
        rightHandSide,integrationRange[points],integratedSolution[points],stepSize
        )

# Plotting the analytic solution, the integrated solution, and their difference
fig, axs = plt.subplots(2, 1, figsize=(10, 5), constrained_layout=True)

ax = axs[0]
ax.plot(integrationRange, analyticSolution(integrationRange), color='darkviolet', label='Analytic Solution')
ax.plot(integrationRange, integratedSolution, color='mediumspringgreen', label='Integrated Solution')
ax.set_ylabel('Solution Value')
ax.grid()
ax.legend()

ax = axs[1]
ax.plot(
    integrationRange,np.abs(analyticSolution(integrationRange)-integratedSolution), color='blue', label='Integrated Error'
    )
ax.set_ylabel('Error')
ax.set_xlabel('x')
ax.grid()
ax.legend()
plt.savefig('PlotI.png',dpi=500)
plt.show()


# %% Defining the fourth order Runge-Kutta function with the two step sizes

# Defining the new RG step function
def rk4_stepd(fun,x,y,h):
    
    # Defining the initial function evaluation
    initialFunctionEvaluation = fun(x,y)
    
    # Performing RK4 for a step size h, this will correspond to y1
    k1 = initialFunctionEvaluation * h
    k2 = fun(x+h/2,y+k1/2) * h
    k3 = fun(x+h/2, y+k2/2) * h
    k4 = fun(x+h, y+k3) * h
    y1 = y+(k1+2*k2+2*k3+k4)/6 
    
    # Peforming RK4 twice for a step size of h/2, these two combined will correspond to y2
    k5 = initialFunctionEvaluation * h/2
    k6 = fun(x+h/4,y+k5/2) * h/2
    k7 = fun(x+h/4, y+k6/2) * h/2
    k8 = fun(x+h/2, y+k7) * h/2
    incompleteY2 = y+(k5+2*k6+2*k7+k8)/6
    
    k9 = fun(x+h/2, incompleteY2) * h/2
    k10 = fun(x+3*h/4, incompleteY2+k9/2) * h/2
    k11 = fun(x+3*h/4, incompleteY2+k10/2) * h/2
    k12 = fun(x+h, incompleteY2+k11) * h/2
    y2 = incompleteY2+(k9+2*k10+2*k11+k12)/6
    
    return y2 + (y2-y1)/15
    
# Once again performing the integration using the step over the asked range
newIntegratedSolution = np.zeros(len(integrationRange))
newIntegratedSolution[0] = 1
stepSize = integrationRange[1] - integrationRange[0]
for points in range(len(integrationRange)-1):
    newIntegratedSolution[points+1] = rk4_stepd(
        rightHandSide,integrationRange[points],newIntegratedSolution[points],stepSize
        )
    
# Plotting the analytic solution, the two integrated solutions, and their difference
fig, axs = plt.subplots(3, 1, figsize=(10, 5), constrained_layout=True)

ax = axs[0]
ax.plot(integrationRange, analyticSolution(integrationRange), color='darkviolet', label='Analytic Solution')
ax.plot(integrationRange, integratedSolution, color='mediumspringgreen', label='Integrated Solution: rk4_step')
ax.plot(integrationRange, newIntegratedSolution, '--', color='blue', label='Integrated Solution: rk4_stepd')
ax.set_ylabel('Solution Value')
ax.grid()
ax.legend()

ax = axs[1]
ax.plot(
    integrationRange,np.abs(analyticSolution(integrationRange)-integratedSolution), color='mediumspringgreen', label='Integrated Error: rk4_step'
    )
ax.set_ylabel('Error')
ax.grid()
ax.legend()

ax = axs[2]
ax.plot(
    integrationRange,np.abs(analyticSolution(integrationRange)-newIntegratedSolution), color='blue', label='Integrated Error: rk4_stepd'
    )
ax.set_ylabel('Error')
ax.set_xlabel('x')
ax.grid()
ax.legend()
plt.savefig('PlotII.png',dpi=500)
plt.show()


# %%
