#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 12:19:25 2025

@author: Matthew
"""
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd

# --- Parameters ---
fs = 44100          # Sampling rate
duration = 2.0      # seconds
n_samples = int(fs * duration)
fc = 500.0  # Center frequency in Hz
bw_frac = 0.05  # Corrected: 5% of center frequency (unitless)  # 5% bandwidth
fwhm = fc * bw_frac  # Corrected FWHM

lowcut = fc * (1 - 0.025)
highcut = fc * (1 + 0.025)

print(f"Band edges: {lowcut:.2f} Hz – {highcut:.2f} Hz")

# --- Step 1: Generate broadband noise ---
noise = np.random.normal(0, 1, n_samples)

# --- FFT of the noise ---
fft_noise = np.fft.rfft(noise)
freqs = np.fft.rfftfreq(n_samples, 1/fs)

# --- Design Gaussian bandpass in frequency domain ---
fwhm = fc * bw_frac  # Corrected FWHM
sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))  # ≈ fwhm / 2.355
gaussian = np.exp(-0.5 * ((freqs - fc)/sigma)**2)

# --- Apply Gaussian bandpass ---
fft_bandpassed = fft_noise * gaussian
bandpassed = np.fft.irfft(fft_bandpassed)

# --- Design complementary notch (1 - Gaussian) ---
fft_notched = fft_noise * (1 - gaussian)
notched = np.fft.irfft(fft_notched)

# --- Normalize before playback ---
bandpassed /= np.max(np.abs(bandpassed))
notched /= np.max(np.abs(notched))


# --- Plot the frequency response ---
plt.figure(figsize=(10, 5))
plt.plot(freqs, gaussian, label='Gaussian Bandpass')
plt.plot(freqs, 1 - gaussian, label='Complementary Notch')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain')
plt.title('Gaussian Filter in Frequency Domain')
plt.legend()
plt.grid(True)
plt.show()


# --- Step 4: Optional playback ---
print("Playing band-passed noise...")
sd.play(bandpassed / np.max(np.abs(bandpassed)), fs)
sd.wait()

print("Playing notch-filtered noise...")
sd.play(notched / np.max(np.abs(notched)), fs)
sd.wait()


# --- Step 6: Dichotic playback tests ---

# Generate a second independent broadband noise for stereo rendering
noise2 = np.random.normal(0, 1, n_samples)
fft_noise2 = np.fft.rfft(noise2)

# Apply same filters to second noise
fft_bandpassed2 = fft_noise2 * gaussian
bandpassed2 = np.fft.irfft(fft_bandpassed2)
bandpassed2 /= np.max(np.abs(bandpassed2))

fft_notched2 = fft_noise2 * (1 - gaussian)
notched2 = np.fft.irfft(fft_notched2)
notched2 /= np.max(np.abs(notched2))

# Stereo A: Bandpassed in both ears
print("Stereo A: Bandpassed L + R")
sd.play(np.stack([bandpassed, bandpassed2], axis=-1), fs)
sd.wait()

# Stereo B: Notch-filtered in both ears
print("Stereo B: Notched L + R")
sd.play(np.stack([notched, notched2], axis=-1), fs)
sd.wait()

# Stereo C: Bandpass Left, Notch Right
print("Stereo C: Bandpass L, Notch R (Pitch illusion)")
sd.play(np.stack([bandpassed, notched2], axis=-1), fs)
sd.wait()

# Stereo D: Notch Left, Bandpass Right
print("Stereo D: Notch L, Bandpass R (Reversed illusion)")
sd.play(np.stack([notched, bandpassed2], axis=-1), fs)
sd.wait()

# --- Step 5: Plot the signals and their spectra ---
plt.figure(figsize=(12, 6))

# Time domain
plt.subplot(2, 1, 1)
plt.plot(bandpassed, label='Band-pass')
plt.plot(notched, label='Notch-filtered', alpha=0.7)
plt.title("Filtered Noise (Time Domain)")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.legend()

# Frequency domain
plt.subplot(2, 1, 2)
freqs = np.fft.rfftfreq(n_samples, 1/fs)
fft_band = np.abs(np.fft.rfft(bandpassed))
fft_notch = np.abs(np.fft.rfft(notched))
plt.semilogy(freqs, fft_band, label='Band-pass')
plt.semilogy(freqs, fft_notch, label='Notch-filtered', alpha=0.7)
plt.title("Filtered Noise (Frequency Domain)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.legend()

plt.tight_layout()
plt.show()


# --- Step 7: Export audio files with timestamped labels ---

import os
from datetime import datetime
from scipy.io.wavfile import write as write_wav

# Create output folder
output_dir = "_audio_renders"
os.makedirs(output_dir, exist_ok=True)

# Create timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Define helper function to save stereo files
def save_wav(label, left, right):
    stereo = np.stack([left, right], axis=-1)
    filename = f"{output_dir}/{timestamp}_{label}.wav"
    write_wav(filename, fs, (stereo * 32767).astype(np.int16))
    print(f"Saved: {filename}")

# Save all combinations
save_wav("stereo_bandpassed", bandpassed, bandpassed2)
save_wav("stereo_notched", notched, notched2)
save_wav("band_left_notch_right", bandpassed, notched2)
save_wav("notch_left_band_right", notched, bandpassed2)
