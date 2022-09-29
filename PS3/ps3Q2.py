#%% Importing modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# %% Defining the decay rate matrix, the solver, and the initial condition

# Defining the half-life matrix of all the products in units of seconds
halfLife = np.asarray([
    4.468*(1e9*365*24*60*60),24.10*(24*60*60),6.70*(60*60),
         245500*(365*24*60*60),75380*(365*24*60*60),1600*(365*24*60*60),
         3.8235*(24*60*60),3.10*(60),26.8*(60),19.9*(60),164.3*(60*1e-6),
         22.3*(365*24*60*60),5.015*(365*24*60*60),138.376*(24*60*60)
])

# Defining the tau matrix
tauMatrix = np.log(2)/halfLife

# Defining a single time step of U238 decay
def solvedProducts(t,γ,rate=tauMatrix):
    
    # Initializing a change in the amount of a product with a change in time (dγ/dt)
    temporalChange = np.zeros(len(rate)+1)
    
    # Computing the change via the RHS of the ODE for t=0
    temporalChange[0] = -γ[0] * rate[0]
    
    # Summing over the changes in each product (excluding Pb206)
    for product in range(1,len(temporalChange)-1):
        temporalChange[product] = γ[product-1]*rate[product-1] - γ[product] * rate[product]

    # Summing over the changes in Pb206
    temporalChange[-1] = γ[-2] * rate[-1]
    
    return temporalChange
    
# Defining the initial condition
initialγ = np.zeros(len(tauMatrix)+1)
initialγ[0] = 1
    
    
# %% Integrating for the given ratios asked in the problem (Pb206 & Th230) & plotting

# Defining the initial & final times and the integration time range for Pb206/U238 and Th230/U234, respectively
pb206Times = [0,10**18]
th230Times = [0,10**13.8]
pb206IntegrationRange = np.logspace(0,np.log10(pb206Times[-1]),1000)
th230IntegrationRange = np.logspace(0,np.log10(th230Times[-1]),1000)

# Defining the solved products for Pb206 and Th230, respectively
solvedPb206 = solve_ivp(solvedProducts,pb206Times,initialγ,method='Radau',t_eval=pb206IntegrationRange)
solvedTh230 = solve_ivp(solvedProducts,th230Times,initialγ,method='Radau',t_eval=th230IntegrationRange)


# Plotting our results, the analytic result, and the difference for Pb206 and Th230, respectively
plt.plot(solvedPb206.t/halfLife[0],solvedPb206.y[-1]/solvedPb206.y[0], color='darkviolet', label='Computed Ratio')
plt.plot(solvedPb206.t/halfLife[0], np.exp(tauMatrix[0]*solvedPb206.t)-1,'--',color='mediumspringgreen', label='Analytic Solution')
plt.ylabel('Ratio Amount')
plt.xlabel(r'$t/t^{U238}_{1/2}$')
plt.grid()
plt.legend()
plt.savefig('PlotIII.png',dpi=500)
plt.show()

plt.plot(solvedTh230.t[1:]/halfLife[0],solvedTh230.y[4][1:]/solvedTh230.y[3][1:], color='darkviolet', label='Computed Ratio')
plt.ylabel('Ratio Amount')
plt.xlabel(r'$t/t^{U234}_{1/2}$')
plt.grid()
plt.legend()
plt.savefig('PlotIV.png',dpi=500)
plt.show()   
    
    
# %%
