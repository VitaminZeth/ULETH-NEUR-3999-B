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
import os
import scipy.io.wavfile as wavfile
import pandas as pd

# --- Parameters ---
fs = 44100          # Sampling rate
duration = 2.0      # seconds
n_samples = int(fs * duration)

# --- Step 0: Create root and subfolders ---
root_folder = os.path.abspath("dichoticpitchstimuli")
os.makedirs(root_folder, exist_ok=True)
print("Root folder path:", root_folder)
folders = {
    "OG-Noise_bounce": os.path.join(root_folder, "OG-Noise_bounce"),
    "gaussian filters": os.path.join(root_folder, "gaussian filters"),
    "delayed_band_info": os.path.join(root_folder, "delayed_band_info"),
    "audio_exports": os.path.join(root_folder, "audio_exports"),
    "origin": os.path.join(root_folder, "origin")
}
for path in folders.values():
    os.makedirs(path, exist_ok=True)

# --- Generate broadband noise ---
noise = np.random.normal(0, 1, n_samples)
# Step 1: export original noise
wavfile.write(os.path.join(folders["OG-Noise_bounce"], "original_noise.wav"), fs, noise.astype(np.float32))

# --- FFT of the noise ---
fft_noise = np.fft.rfft(noise)
freqs = np.fft.rfftfreq(n_samples, 1/fs)
# Step 2: save FFT data
amplitude_dbfs = 20 * np.log10(np.abs(fft_noise) / np.max(np.abs(fft_noise)))
phase = np.angle(fft_noise)
df_noise = pd.DataFrame({
    "Frequency (Hz)": freqs,
    "Amplitude_dBFS": amplitude_dbfs,
    "Phase_radians": phase
})
df_noise.to_csv(os.path.join(folders["OG-Noise_bounce"], "og_noise_csv.csv"), index=False)

# --- Design Gaussian bandpass in frequency domain ---
M = len(fft_noise)
alpha = 200
std = (M - 1) / (2 * alpha)
window = gaussian(M, std)
mid = len(freqs)//2

# --- Assigning the centre frequency ---
bandCentre1 = 1000
bandCentre2 = 1000

# --- Create windows and plot bandpasses ---
window1 = np.roll(window, bandCentre1-mid)
window2 = np.roll(window, bandCentre2-mid)
plt.plot(freqs, window1)
plt.plot(freqs, window2)

# --- Create complementary windows ---
complementaryWindow = 1 - gaussian(M, std)
complementaryWindow1 = np.roll(complementaryWindow, bandCentre1-mid)
plt.plot(freqs, complementaryWindow1)
complementaryWindow2 = np.roll(complementaryWindow, bandCentre2-mid)
plt.plot(freqs, complementaryWindow2)

# Step 3: save Gaussian filter data and plot
pd.DataFrame({
    "Frequency (Hz)": freqs,
    "Window1 Gain": window1,
    "Window2 Gain": window2
}).to_csv(os.path.join(folders["gaussian filters"], "gaussian_filters.csv"), index=False)
plt.figure()
plt.plot(freqs, window1, label='Window1')
plt.plot(freqs, window2, label='Window2')
plt.title("Gaussian Filter Responses")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Gain")
plt.legend()
plt.savefig(os.path.join(folders["gaussian filters"], "gaussian_filters.png"))
plt.close()

# --- Apply Gaussian bandpass ---
fft_bandpassed1 = fft_noise * window1
bandpassed1 = np.fft.irfft(fft_bandpassed1)

# --- Apply shifted Gaussian bandpass ---
fft_bandpassed2 = fft_noise * window2
bandpassed2 = np.fft.irfft(fft_bandpassed2)

# --- Apply a sample delay ---
delayed_bandpassed2 = np.roll(bandpassed2, 7)
print(f"Delayed signal (delay integer shift): {delayed_bandpassed2}")

# Step 4: save delayed FFT data and plot
fft_delayed = np.fft.rfft(delayed_bandpassed2)
amplitude_delayed = 20 * np.log10(np.abs(fft_delayed) / np.max(np.abs(fft_delayed)))
phase_delayed = np.angle(fft_delayed)
df_delayed = pd.DataFrame({
    "Frequency (Hz)": freqs,
    "Amplitude_dBFS": amplitude_delayed,
    "Phase_radians": phase_delayed
})
df_delayed.to_csv(os.path.join(folders["delayed_band_info"], "delayed_bandpassed2.csv"), index=False)
plt.figure()
plt.plot(freqs, amplitude_delayed)
plt.title("FFT of delayed_bandpassed2")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude (dBFS)")
plt.savefig(os.path.join(folders["delayed_band_info"], "delayed_fft_plot.png"))
plt.close()

# --- Design complementary notch ---
fft_notched1 = fft_noise * complementaryWindow1
notched1 = np.fft.irfft(fft_notched1)
fft_notched2 = fft_noise * complementaryWindow2
notched2 = np.fft.irfft(fft_notched2)

# --- Blend the files ---
mixedl = bandpassed1 + notched1
mixedr = delayed_bandpassed2 + notched2

# --- Combine into a stereo array ---
mixed = np.column_stack((mixedl, mixedr))

# --- Optional playback and exports ---
print("Playing band-passed noise...")
# export before playback
wavfile.write(os.path.join(folders["audio_exports"], "bandpassed1.wav"), fs, bandpassed1.astype(np.float32))
sd.play(bandpassed1 / np.max(np.abs(bandpassed1)), fs)
sd.wait()

print("Playing shifted band-passed noise...")
wavfile.write(os.path.join(folders["audio_exports"], "delayed_bandpassed2.wav"), fs, delayed_bandpassed2.astype(np.float32))
sd.play(delayed_bandpassed2 / np.max(np.abs(delayed_bandpassed2)), fs)
sd.wait()

print("Playing notch-filtered noise...")
wavfile.write(os.path.join(folders["audio_exports"], "notched1.wav"), fs, notched1.astype(np.float32))
sd.play(notched1 / np.max(np.abs(notched1)), fs)
sd.wait()

print("Playing shifted notch-filtered noise...")
wavfile.write(os.path.join(folders["audio_exports"], "notched2.wav"), fs, notched2.astype(np.float32))
sd.play(notched2 / np.max(np.abs(notched2)), fs)
sd.wait()

print("Playing Mixed Left Channel...")
wavfile.write(os.path.join(folders["audio_exports"], "mixedl.wav"), fs, mixedl.astype(np.float32))
sd.play(mixedl / np.max(np.abs(mixedl)), fs)
sd.wait()

print("Playing Mixed Right Channel...")
wavfile.write(os.path.join(folders["audio_exports"], "mixedr.wav"), fs, mixedr.astype(np.float32))
sd.play(mixedr / np.max(np.abs(mixedr)), fs)
sd.wait()

print("Playing Mixed L + R... 1")
wavfile.write(os.path.join(folders["origin"], "final_mix.wav"), fs, (mixed / np.max(np.abs(mixed))).astype(np.float32))
sd.play(mixed / np.max(np.abs(mixed)), fs)
sd.wait()

print("Playing Mixed L + R... 2")
sd.play(mixed / np.max(np.abs(mixed)), fs)
sd.wait()

print("Playing Mixed L + R... 3")
sd.play(mixed / np.max(np.abs(mixed)), fs)
sd.wait()

print("Playing Mixed L + R... 4")
sd.play(mixed / np.max(np.abs(mixed)), fs)
sd.wait()
