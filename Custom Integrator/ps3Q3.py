#%% Importing modules
import numpy as np

# %% Loading the text file, defining the points on the difference axes, and defining our new parameters

# Defining the multi-axis points
xAxis, yAxis, zAxis = np.loadtxt('./dish_zenith.txt').T

# Defining <d> = Am: first initializing a matrix, then defining the parameter coefficients
Am = np.zeros((len(xAxis),4))
Am[:,0]=xAxis**2+yAxis**2
Am[:,1]=xAxis
Am[:,2]=yAxis
Am[:,3]=np.ones(len(xAxis))


# %% Solving the system for m and computed the parameters

# Explicitly solving for m using the linear least squares model
solvedM = np.linalg.inv(Am.T@Am)@Am.T@zAxis

# Redefining the parameters we wished to solve for before linearizing
a = solvedM[0]
x0=solvedM[1]/(-2*a)
y0=solvedM[2]/(-2*a)
z0=solvedM[3]-a*(x0**2+y0**2)

# Printing the output of the best fit parameters
print('The following are the best fit parameters [x0,y0,z0,a]: ',x0,y0,z0,a)

# %% Computing the noise matrix, the uncertainty in a, the focal length, and the focal length error bar

# Constructing the noise matrix
noiseMatrixComponent = zAxis - Am@solvedM
noiseMatrix = np.diag(noiseMatrixComponent)

# Computing the covariance matrix and outputting the uncertainty in a
covarianceMatrix = np.linalg.inv(Am.T@np.linalg.inv(noiseMatrix)@Am) 
print('The following is the uncertainty in a: ', np.sqrt(covarianceMatrix[0,0]))

# Computing the focal length and its error bar
focalLength = 1/(4*a)
focalError = np.sqrt(covarianceMatrix[0,0])/(4*a**2) 
print('The following is the focal length and its error: ', focalLength, focalError)


# %%
