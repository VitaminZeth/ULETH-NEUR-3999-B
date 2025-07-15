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
# sd.default.samplerate = fs # Set default sd Sampling rate
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
bandCentre1 = 1000
bandCentre2 = 1000

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
fft_bandpassed1 = fft_noise * window1 # Freq Domain
bandpassed1 = np.fft.irfft(fft_bandpassed1) #plot the points

# --- Apply a shifted Gaussian bandpass to a copy
fft_bandpassed2 = fft_noise * window2 # Freq Domain
bandpassed2 = np.fft.irfft(fft_bandpassed2) #plot the points

# --- Apply a sample delay to the shifted Gaussian bandpassed copy

# # Define the delay in samples
# delay_samples = 10  # Adjust as needed

# # Create a zero-padded array for the delay
# delayed_bandpassed2 = np.pad(bandpassed2, (delay_samples, 0), 'constant')
# ValueError: operands could not be broadcast together with shapes (88210,) (88200,) 

# Shift by 2 samples to the right (delay) 
delayed_bandpassed2 = np.roll(bandpassed2, 7)
print(f"Delayed signal (delay integer shift): {delayed_bandpassed2}")

# Shift by 2 samples to the left (advance)
# advanced_bandpassed2 = np.roll(signal, -2)
# print(f"Advanced signal (advance integer shift): {advanced_signal}")

# --- Design complementary notch (1 - Gaussian) ---
fft_notched1 = fft_noise * complementaryWindow1 # Freq Domain
notched1 = np.fft.irfft(fft_notched1) #plot the points

# --- Design complementary shifted notch (1 - Gaussian) ---
fft_notched2 = fft_noise * complementaryWindow2 # Freq Domain
notched2 = np.fft.irfft(fft_notched2) #plot the points

# --- Blend the files
mixedl = (bandpassed1 + notched1)
mixedr = (delayed_bandpassed2 + notched2)
                                            
# --- Combine into a stereo array (two columns)                                            
mixed = np.column_stack((mixedl, mixedr))

# --- Optional playback ---

# --- Play the band-passed
print("Playing band-passed noise...")
sd.play(bandpassed1 / np.max(np.abs(bandpassed1)), fs)
sd.wait()

print("Playing shifted band-passed noise...")
sd.play(delayed_bandpassed2 / np.max(np.abs(delayed_bandpassed2)), fs)
sd.wait()

# --- Play the notched
print("Playing notch-filtered noise...")
sd.play(notched1 / np.max(np.abs(notched1)), fs)
sd.wait()

# --- Play the shifted notched
print("Playing shifted notch-filtered noise...")
sd.play(notched2 / np.max(np.abs(notched2)), fs)
sd.wait()

# --- Play the notched and band-passed together (mono)
print("Playing Mixed Left Channel...")
sd.play(mixedl / np.max(np.abs(mixedl)), fs)
sd.wait()

# --- Play the notched and band-passed together (mono)
print("Playing Mixed Right Channel...")
sd.play(mixedr / np.max(np.abs(mixedr)), fs)
sd.wait()

print("Playing Stimuli 4 Times...")

# --- Play the notched and band-passedtogether (stereo)
print("Playing Mixed L + R... 1")
sd.play(mixed / np.max(np.abs(mixed)), fs)
sd.wait()

# --- Play the notched and band-passedtogether (stereo)
print("Playing Mixed L + R... 2")
sd.play(mixed / np.max(np.abs(mixed)), fs)
sd.wait()

# --- Play the notched and band-passedtogether (stereo)
print("Playing Mixed L + R... 3")
sd.play(mixed / np.max(np.abs(mixed)), fs)
sd.wait()

# --- Play the notched and band-passedtogether (stereo)
print("Playing Mixed L + R... 4")
sd.play(mixed / np.max(np.abs(mixed)), fs)
sd.wait()

# NOTE TO SELF:
#   Everythign worked out for the copy, but I need to fix it so that the playback of all the individual files works
#   as well as the notch staying the same, and only moving the band-pass
