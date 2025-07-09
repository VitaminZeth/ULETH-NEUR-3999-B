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
sd.default.samplerate = fs # Set default sd Sampling rate
duration = 2.0      # seconds
n_samples = int(fs * duration)

# --- Generate broadband noise ---
noise = np.random.normal(0, 1, n_samples)

# --- FFT of the noise ---
fft_noise = np.fft.rfft(noise)
freqs = np.fft.rfftfreq(n_samples, 1/fs)

# --- Design Gaussian bandpass in frequency domain ---
M =   len(fft_noise)          # Length of window (odd number is common)
alpha = 200
std =  (M-1)/(2*alpha)     # Shape parameter (analogous to 'alpha' in MATLAB gausswin)

# --- Create the window (band-passed)
window = gaussian(M, std)  #a  std of 1000 will be full-width half max of ....
mid=len(freqs)//2

# --- Assigning the centre freqeuncy
bandCentre1 = 5000
bandCentre2 = 6000

# --- Plotting the two bandpasses
window1 = np.roll(window,bandCentre1-mid)
plt.plot(freqs,window1)

window2 = np.roll(window,bandCentre2-mid)
plt.plot(freqs,window2)

# --- Create the complementary window (notched)
complementaryWindow = 1-gaussian(M, std)

complementaryWindow1 = np.roll(complementaryWindow,bandCentre1-mid)
plt.plot(freqs, complementaryWindow1) #plot the points

# --- Create a copy of the complementary window (notched)
complementaryWindow2 = np.roll(complementaryWindow,bandCentre2-mid)
plt.plot(freqs, complementaryWindow2) #plot the points

# --- Apply Gaussian bandpass
fft_bandpassed1 = fft_noise * window1
bandpassed1 = np.fft.irfft(fft_bandpassed1)

# --- Apply a shifted Gaussian bandpass to a copy
fft_bandpassed2 = fft_noise * window2
bandpassed2 = np.fft.irfft(fft_bandpassed2)

# --- Design complementary notch (1 - Gaussian) ---
fft_notched1 = fft_noise * complementaryWindow1
notched1 = np.fft.irfft(fft_notched1) #plot the points

# --- Design complementary shifted notch (1 - Gaussian) ---
fft_notched2 = fft_noise * complementaryWindow2
notched2 = np.fft.irfft(fft_notched2) #plot the points

# --- Blend the files
mixedl = (bandpassed1+notched1)
mixedr = (bandpassed2+notched2)
                                            
# --- Combine into a stereo array (two columns)                                            
mixed = np.column_stack((mixedl, mixedr))

# --- Optional playback ---

# --- Play the band-passed
print("Playing band-passed noise...")
sd.play(bandpassed1 / np.max(np.abs(bandpassed1)))
sd.wait()

print("Playing shifted band-passed noise...")
sd.play(bandpassed2 / np.max(np.abs(bandpassed2)))
sd.wait()

# --- Play the notched
print("Playing notch-filtered noise...")
sd.play(notched1 / np.max(np.abs(notched1)))
sd.wait()

# --- Play the shifted notched
print("Playing notch-filtered noise...")
sd.play(notched2 / np.max(np.abs(notched2)))
sd.wait()

# # --- Play the notched and band-passed together (mono)
# print("Playing Left Channel...")
# sd.play(mixedl / np.max(np.abs(mixedl)))
# sd.wait()

# # --- Play the notched and band-passed together (mono)
# print("Playing Right Channel...")
# sd.play(mixedr / np.max(np.abs(mixedr)))
# sd.wait()

# # --- Play the notched and band-passedtogether (stereo)
# print("Playing MIXED L+R...")
# sd.play(mixed / np.max(np.abs(mixed)))
# sd.wait()


