#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 12:19:25 2025

@author: Matthew
"""
import numpy as np
from scipy.signal.windows import gaussian
import matplotlib.pyplot as plt
import sounddevice as sd

# --- Parameters ---
fs = 44100          # Sampling rate
duration = 2.0      # seconds
n_samples = int(fs * duration)




# --- Step 1: Generate broadband noise ---
noise = np.random.normal(0, 1, n_samples)

# --- FFT of the noise ---
fft_noise = np.fft.rfft(noise)
freqs = np.fft.rfftfreq(n_samples, 1/fs)

# --- Design Gaussian bandpass in frequency domain ---
M =   len(fft_noise)          # Length of window (odd number is common)
alpha = 200.5
std =  (M-1)/(2*alpha)     # Shape parameter (analogous to 'alpha' in MATLAB gausswin)

# Create the window
window = gaussian(M, std)  #a  std of 1000 will be full-width half max of ....
mid=len(freqs)//2
bandCentre = 5000

window = np.roll(window,bandCentre-mid)

plt.plot(freqs,window)



complementaryWindow = 1-gaussian(M, std)

complementaryWindow= np.roll(complementaryWindow,bandCentre-mid)
plt.plot(freqs, complementaryWindow)




# --- Apply Gaussian bandpass ---
fft_bandpassed = fft_noise * window
bandpassed = np.fft.irfft(fft_bandpassed)

# --- Design complementary notch (1 - Gaussian) ---
fft_notched = fft_noise * complementaryWindow
notched = np.fft.irfft(fft_notched)

mixed = bandpassed+notched



# --- Step 4: Optional playback ---
print("Playing band-passed noise...")
sd.play(bandpassed / np.max(np.abs(bandpassed)), fs)
sd.wait()

print("Playing notch-filtered noise...")
sd.play(notched / np.max(np.abs(notched)), fs)
sd.wait()

print("Playing notch-filtered noise...")
sd.play(mixed / np.max(np.abs(mixed)), fs)
sd.wait()
