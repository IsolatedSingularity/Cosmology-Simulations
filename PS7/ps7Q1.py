#%% Importing modules
import numpy as np
import matplotlib.pyplot as plt


# %% Extracting the values of the randomly generated points and plotting it

# Extracting the 3-tulpe values from the generated random points
threeTuples = np.transpose(np.loadtxt("rand_points.txt"))
xPositions = threeTuples[0]
yPositions = threeTuples[1]
zPositions = threeTuples[2]

# Plotting along the fixed plane z = 2x - y
plt.figure( figsize=(10,8) )
plt.scatter(2*xPositions - yPositions,zPositions, color='darkviolet', label='Plane Tuples')
plt.xlabel('2x-y')
plt.ylabel('z')
plt.grid()
plt.legend()
plt.savefig('PlotI.png',dpi=500)


# %% Randomly producing 3-tuples in Python and plotting the same planar restriction

# Generating random 3-tuples
generatedTuples = np.random.rand(
    len(xPositions),3
)
generatedXPositions = generatedTuples.T[0]
generatedYPositions = generatedTuples.T[1]
generatedZPositions = generatedTuples.T[2]

# Plotting along the fixed plane z = 2x - y
plt.figure( figsize=(10,8) )
plt.scatter(
    2*generatedXPositions - generatedYPositions,generatedZPositions, color='darkviolet', label='Plane Tuples'
    )
plt.xlabel('2x-y')
plt.ylabel('z')
plt.grid()
plt.legend()
plt.savefig('PlotII.png',dpi=500)


# %%
