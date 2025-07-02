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
bw_frac = 0.05 * fc  # 5% bandwidth

lowcut = fc * (1 - 0.025)
highcut = fc * (1 + 0.025)

print(f"Band edges: {lowcut:.2f} Hz – {highcut:.2f} Hz")

# --- Step 1: Generate broadband noise ---
noise = np.random.normal(0, 1, n_samples)

# --- FFT of the noise ---
fft_noise = np.fft.rfft(noise)
freqs = np.fft.rfftfreq(n_samples, 1/fs)

# --- Design Gaussian bandpass in frequency domain ---
fwhm = fc * bw_frac
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
