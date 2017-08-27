"""
===================
Peak FFT frequency
===================

"""

import numpy as np
from scipy import fftpack
from matplotlib import pyplot as plt

# Generate the signal
# -------------------

# Seed the random number generator
np.random.seed(1234)

time_step = 0.02
period = 5.

time_vec = np.arange(0, 20, time_step)
sig = (np.sin(2 * np.pi / period * time_vec)
       + 0.5 * np.random.randn(time_vec.size))

# Compute and plot the power and zoom around the peak
# ---------------------------------------------------

sample_freq = fftpack.fftfreq(sig.size, d=time_step)
sig_fft = fftpack.fft(sig)
pidxs = np.where(sample_freq > 0)
freqs = sample_freq[pidxs]
power = np.abs(sig_fft)[pidxs]

# Plot the FFT power
plt.plot(freqs, power)
plt.xlabel('Frequency [Hz]')
plt.ylabel('plower')

freq = freqs[power.argmax()]

# An inner plot to show the peak frequency
axes = plt.axes([0.3, 0.3, 0.5, 0.5])
plt.title('Peak frequency')
plt.plot(freqs[:8], power[:8])
plt.setp(axes, yticks=[])

plt.show()
