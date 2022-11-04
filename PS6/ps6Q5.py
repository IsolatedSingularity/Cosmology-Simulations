#%% Importing modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c
import ligoReader as ligo
import json


# %% Defining the working directory and extracting the data

# Defining the working directory
workingDirectory = './LOSC_Event_tutorial'

# Defining the events within the data
dataEvents = json.load(
    open(
       workingDirectory +'/BBH_events_v3.json','r' 
    )
)

# Initializing seperate empty dictionaries for the two detectors
hanfordData = {}
livingstonData = {}

# Extracting parameters for both detectors by looping over the data
for event in dataEvents:
    
    # Hanford detector parameters
    strain_H, time_H, chan_dict_H = ligo.read_file(workingDirectory +'/'+ dataEvents[event]['fn_H1'])
    hanfordData[event]={'strain':strain_H,'time':time_H,'chan_dict_H':chan_dict_H}
    
    # Livingston detector parameters
    strain_L, time_L, chan_dict_L = ligo.read_file(workingDirectory +'/'+ dataEvents[event]['fn_L1'])
    livingstonData[event]={'strain':strain_L,'time':time_L,'chan_dict_H':chan_dict_L}
    

# %% Computing SFT and NFT, then smoothing the power spectrum with a window function

# All of this is done iteratively for each event in the data
for event in dataEvents:

    # Smoothing the spectrum with the window function
    hanfordSize = len(hanfordData[event]['strain'])
    windowedHanford = ligo.make_flat_window(hanfordSize, hanfordSize//2)
    
    livingstonSize = len(livingstonData[event]['strain'])
    windowedLivingstone = ligo.make_flat_window(livingstonSize, livingstonSize//2)
    
    # Computing SFT and NFT for the windowed data
    hanfordData[event]['SFT'] = np.fft.rfft(
        hanfordData[event]['strain']*windowedHanford
        )
    hanfordData[event]['NFT'] = np.abs(hanfordData[event]['SFT'])**2
    
    livingstonData[event]['SFT'] = np.fft.rfft(
        livingstonData[event]['strain']*windowedLivingstone
        )
    livingstonData[event]['NFT'] = np.abs(livingstonData[event]['SFT'])**2 
    
    # Averaging over the noise (smoothing)
    for iteration in range(10): 
        
        hanfordFT = hanfordData[event]['NFT'] 
        hanfordData[event]['NFT'] = (hanfordFT+np.roll(hanfordFT,1)+np.roll(hanfordFT,-1))/3
        
        livingstonFT = livingstonData[event]['NFT']
        livingstonData[event]['NFT'] = (livingstonFT+np.roll(livingstonFT,1)+np.roll(livingstonFT,-1))/3
        

# %% Plotting all the noise estimates for each event
fig, axs = plt.subplots(2, 2, figsize=(10, 5), constrained_layout=True)

ax = axs[0,0]
ax.loglog(hanfordData['GW150914']['NFT'],label='Hanford NFT', color='mediumspringgreen')
ax.loglog(livingstonData['GW150914']['NFT'],label='Livingstone NFT', color='darkviolet')
ax.grid()
ax.legend()
ax.set_title('GW150914')

ax = axs[1,0]
ax.loglog(hanfordData['LVT151012']['NFT'],label='Hanford NFT', color='mediumspringgreen')
ax.loglog(livingstonData['LVT151012']['NFT'],label='Livingstone NFT', color='darkviolet')
ax.grid()
ax.legend()
ax.set_title('LVT151012')

ax = axs[0,1]
ax.loglog(hanfordData['GW151226']['NFT'],label='Hanford NFT', color='mediumspringgreen')
ax.loglog(livingstonData['GW151226']['NFT'],label='Livingstone NFT', color='darkviolet')
ax.grid()
ax.legend()
ax.set_title('GW151226')

ax = axs[1,1]
ax.loglog(hanfordData['GW170104']['NFT'],label='Hanford NFT', color='mediumspringgreen')
ax.loglog(livingstonData['GW170104']['NFT'],label='Livingstone NFT', color='darkviolet')
ax.grid()
ax.legend()
ax.set_title('GW170104')

plt.savefig('PlotVII.png', dpi=500)


# %% Defining the templates, prewhittening the templates + strain, and performing the match filter

# All of this is done iteratively for each event in the data
for event in dataEvents:
    
    # Extracting the signal templates from the data
    hanfordData[event]['template'], livingstonData[event]['template'] = ligo.read_template(
        workingDirectory +'/'+ dataEvents[event]['fn_template']
        )
    hanfordData[event]['TFT'] = np.abs(
        np.fft.rfft(hanfordData[event]['template'])
        )
    livingstonData[event]['TFT'] = np.abs(
        np.fft.rfft(livingstonData[event]['template'])
        )
    
    # Prewhittening the strain and templates prior to the cross correlation
    hanfordData[event]['SFT_white'] = hanfordData[event]['SFT']/np.sqrt(hanfordData[event]['NFT'])
    hanfordData[event]['TFT_white'] = np.fft.rfft(hanfordData[event]['template']*windowedHanford)/np.sqrt(hanfordData[event]['NFT'])
    
    livingstonData[event]['SFT_white'] = livingstonData[event]['SFT']/np.sqrt(livingstonData[event]['NFT'])
    livingstonData[event]['TFT_white'] = np.fft.rfft(livingstonData[event]['template']*windowedLivingstone)/np.sqrt(livingstonData[event]['NFT'])
    
    # Performing the match filter (cross correlation) with FFTs
    hanfordData[event]['MF'] = np.fft.irfft(hanfordData[event]['SFT_white']*np.conj(hanfordData[event]['TFT_white']))
    livingstonData[event]['MF'] = np.fft.irfft(livingstonData[event]['SFT_white']*np.conj(livingstonData[event]['TFT_white']))
    
    
# %% Plotting all the match filters for each event
fig, axs = plt.subplots(2, 2, figsize=(10, 5), constrained_layout=True)

ax = axs[0,0]
ax.plot(hanfordData['GW150914']['MF'],label='Hanford Matched Filter', color='mediumspringgreen')
ax.plot(livingstonData['GW150914']['MF'],label='Livingstone Matched Filter', color='darkviolet')
ax.grid()
ax.legend()
ax.set_title('GW150914')

ax = axs[1,0]
ax.plot(hanfordData['LVT151012']['MF'],label='Hanford Matched Filter', color='mediumspringgreen')
ax.plot(livingstonData['LVT151012']['MF'],label='Livingstone Matched Filter', color='darkviolet')
ax.grid()
ax.legend()
ax.set_title('LVT151012')

ax = axs[0,1]
ax.plot(hanfordData['GW151226']['MF'],label='Hanford Matched Filter', color='mediumspringgreen')
ax.plot(livingstonData['GW151226']['MF'],label='Livingstone Matched Filter', color='darkviolet')
ax.grid()
ax.legend()
ax.set_title('GW151226')

ax = axs[1,1]
ax.plot(hanfordData['GW170104']['MF'],label='Hanford Matched Filter', color='mediumspringgreen')
ax.plot(livingstonData['GW170104']['MF'],label='Livingstone Matched Filter', color='darkviolet')
ax.grid()
ax.legend()
ax.set_title('GW170104')

plt.savefig('PlotVIII.png', dpi=500)


# %% Computing the signal to noise ratio (SNR) for each event, both for seperate and combined detector data

# Defining the branch cut for which the data isn't suppressed by the window function
branchCut = 40000

# Creating a new array for combined detector event data
combinedEvents = dataEvents

# All of this is done iteratively for each event in the data
for event in dataEvents:
    
    # Computing the SNR via the quotient of the maximum of the match filter to the cut noise quotient (std of the match filter)
    hanfordData[event]['SNR'] = np.max(
        np.abs(hanfordData[event]['MF']/np.std(hanfordData[event]['MF'][:branchCut]))
        )
    livingstonData[event]['SNR'] = np.max(
        np.abs(livingstonData[event]['MF']/np.std(livingstonData[event]['MF'][:branchCut]))
        )
    
    # Computing the SNR for the combined detector data sets   
    combinedEvents[event]['SNR'] = np.sqrt(
        hanfordData[event]['SNR']**2 + livingstonData[event]['SNR']**2
        )

    # Printing the results of the SNR analysis
    print(event,': Hanford SNR =', hanfordData[event]['SNR'],', Livingston SNR =', livingstonData[event]['SNR'],
         ' Combined SNR = ', combinedEvents[event]['SNR'],'\n')


# %%
