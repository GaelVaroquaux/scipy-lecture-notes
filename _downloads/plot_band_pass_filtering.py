"""
=============================================
Band-pass filtering with FFT and windows
=============================================

This example demonstrate :func:`scipy.fftpack.fft`,
:func:`scipy.fftpack.fftfreq` and :func:`scipy.fftpack.ifft`. It
implements a basic filter that is very suboptimal, and should not be
used.

"""

import numpy as np
from scipy import fftpack
from matplotlib import pyplot as plt

############################################################
# Generate the signal
############################################################

# Seed the random number generator
np.random.seed(1234)

time_step = 0.02
time_vec = np.arange(0, 20, time_step)

# A signal with a small frequency chirp
sig = np.sin(0.4 * np.pi * time_vec * (1 + .1 * time_vec))

noisy_sig = sig + 0.4 * np.random.randn(time_vec.size)

plt.figure(figsize=(6, 5))
plt.plot(time_vec, noisy_sig, label='Noisy signal')
plt.plot(time_vec, sig, linewidth=3, label='Noiseless signal')

plt.legend(loc='best')

############################################################
# Compute and plot the power
############################################################

# The FFT of the signal
sig_fft = fftpack.fft(noisy_sig)

# And the power (sig_fft is of complex dtype)
power = np.abs(sig_fft)

# The corresponding frequencies
sample_freq = fftpack.fftfreq(sig.size, d=time_step)

# Plot the FFT power
plt.figure(figsize=(6, 5))
plt.plot(sample_freq, power)
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power')

# Find the peak frequency: we can focus on only the positive frequencies
pos_mask = np.where(sample_freq > 0)
freqs = sample_freq[pos_mask]
peak_idx = power[pos_mask].argmax()
peak_freq = freqs[peak_idx]

# An inner plot to show the peak frequency
axes = plt.axes([0.55, 0.3, 0.3, 0.5])
plt.title('Aroud the peak\nfrequency')
plt.plot(freqs[:25], power[:25])
plt.axvline(peak_freq, color='gray')
plt.axis('tight')
plt.setp(axes, yticks=[])

# scipy.signal.find_peaks_cwt can also be used for more advanced
# peak detection

############################################################
# Remove all the high frequencies
############################################################
#
# We now remove all the high frequencies and transform back from
# frequencies to signal.

high_freq_fft = sig_fft.copy()
high_freq_fft[np.abs(sample_freq) > peak_freq] = 0
filtered_sig = fftpack.ifft(high_freq_fft)

plt.figure(figsize=(6, 5))
plt.plot(time_vec, filtered_sig, linewidth=3,
         label='Noisy signal without high freqs')
plt.plot(time_vec, sig, label='Original signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

plt.legend(loc='best')

############################################################
# **Note** Note how the filtering has destroyed the signal: some of the
# high-frequencies where part of the signal and not the noise.
#
# Let us build a bandpass filter

############################################################
# Naive filtering around the peak frequency
############################################################
#
# Let us keep only the frequencies close to that of the signal of
# interest.
band_fft = sig_fft.copy()
band_fft[np.abs(np.abs(sample_freq) - peak_freq) > 1] = 0
band_sig = fftpack.ifft(band_fft)

plt.figure(figsize=(6, 5))
plt.plot(time_vec, band_sig, linewidth=3, label='Band-cut noisy signal')
plt.plot(time_vec, sig, linewidth=2, label='Original signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

plt.legend(loc='best')

############################################################
# Still a bit noisy.
#
# Well, filter design is hard. Maybe we didn't should the best cut-off.


############################################################
# Better band filtering around the peak frequency
############################################################
#
# How we filter around the peak, ie the window used, is actually crucial.
# We will now use a good window, with an optimized fall-off, rather than
# a hard cut-off.

# Create an hamming window:
from scipy import signal
window = signal.get_window('hamming', 2 * peak_idx)

windowed_fft = sig_fft.copy()
windowed_fft[np.abs(np.abs(sample_freq) - peak_freq) > 1] *= 0
windowed_sig = fftpack.ifft(windowed_fft)

plt.figure(figsize=(6, 5))
plt.plot(time_vec, windowed_sig, linewidth=3, label='Band-pass with window')
plt.plot(time_vec, sig, label='Original signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

plt.legend(loc='best')


############################################################
# Once again, the signal is distorted.

plt.show()


