import numpy as np
import scipy as sp
from scipy import signal
import matplotlib.pyplot as plt

# Read in force and EMG data
infile = open('mvc.lvm', 'r')
line = infile.readlines()[23:]
infile.close()

data = [row.strip().split(',') for row in line]

time, force, emg = [], [], []
time = np.array([float(row[0]) for row in data])
force = np.array([float(row[1]) for row in data])
emg = np.array([float(row[2]) for row in data])

# Write a function to process EMG, and process the EMG signal
sampling_rate = 2000
def getEMG(emg, sampling_rate, highpass=20, lowpass=450):
    # remove mean
    emg = emg - np.mean(emg)
    # create filter parameters and digitally filter signal
    high, low = highpass/sampling_rate, lowpass/sampling_rate
    b, a = sp.signal.butter(4, [high, low], btype='bandpass')
    emg = sp.signal.filtfilt(b, a, emg)
    # take absolute of emg
    emg = abs(emg)
    return emg

emg = getEMG(emg, 2000)

# Find MVC force, mark corresponding EMG and 50 ms window across it on graph
force_mvc = np.max(force)
index_mvc = np.where(force == force_mvc)[0]
index_mvc = int(index_mvc)
# find no. of samples over 25 ms
width = int(0.025 * sampling_rate)
# find mean EMG and highest possible EMG voltage
emg_mvc = np.mean(emg[index_mvc-width : index_mvc+width])
emg_max = np.max(emg)
print('MVC EMG over 50 ms (a.u.): {:.2f}'.format(emg_mvc))

# Have user select baseline EMG region
plt.plot(time, emg)
x = plt.ginput(2)
print('Start and stop x-values: {:.2f}, {:.2f}'.format(x[0][0], x[1][0]))
plt.show()
plt.close()

# Find index values for start and stop points
start_time = x[0][0]
stop_time = x[1][0]
for i,j in enumerate(time):
    if j == np.around(start_time, decimals=2): # round to 2 dec pla
        start_index = i
    if j == np.around(stop_time, decimals=2):
        stop_index = i
print('Start and stop index values: {:d}, {:d}'.format(start_index, stop_index))

# Plot force and EMG on time
fig = plt.figure()
plt.clf()
plt.subplot(2,1,1)
plt.plot(time, force)
plt.xlabel('Time')
plt.ylabel('Force (V)')
plt.subplot(2, 1, 2)
plt.plot(time, emg)
plt.xlabel('Time')
plt.ylabel('EMG (a.u.)')

# Mark baseline and MVC regions on plot
x1, x2 = time[index_mvc-width], time[index_mvc+width]
y1, y2 = emg_max+0.01, emg_max+0.01
plt.plot([x1, x2], [y1, y2], color='r', linestyle='-', linewidth=5,)
plt.plot(time[start_index:stop_index], emg[start_index:stop_index], color='g')

# Format figure spacing and save
fig.tight_layout()
fig.savefig('mvc.png')
