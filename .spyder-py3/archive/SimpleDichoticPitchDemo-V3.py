#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 12:19:25 2025

@author: Seth Villamil & ChatGPT
"""
import numpy as np
from scipy.signal.windows import gaussian
import matplotlib.pyplot as plt
import sounddevice as sd

# --- Parameters ---
fs = 44100          # Sampling rate
duration = 1.0      # seconds
n_samples = int(fs * duration)

# --- Centre frequency for pitch illusion ---
bandCentre = 500  # Hz

# --- Desired filter bandwidth: 1/20th of center frequency ---
bandwidth_hz = bandCentre / 20.0
bin_width_hz = fs / n_samples
bandwidth_bins = bandwidth_hz / bin_width_hz
std = bandwidth_bins / 2.355  # Convert FWHM to std (bins)

# --- Volume controls ---
bandpass_volume = 7.0
background_volume = 0.5

# --- Generate broadband noise ---
noise = np.random.normal(0, 1, n_samples)

# --- FFT of the noise ---
fft_noise = np.fft.rfft(noise)
freqs = np.fft.rfftfreq(n_samples, 1/fs)

# --- Create Gaussian window ---
M = len(fft_noise)
window = gaussian(M, std)

# --- Shift Gaussian to the desired center frequency ---
shift_bins = int(round(bandCentre / bin_width_hz))
window_shifted = np.roll(window, shift_bins)

# --- Extract band-limited portion ---
fft_band = fft_noise * window_shifted

# --- Phase shift that band by 180 degrees ---
fft_band_shifted = fft_band * np.exp(1j * np.pi)

# --- Build left and right ear spectra ---
fft_left = fft_noise
fft_right = (fft_noise * (1 - window_shifted)) + fft_band_shifted

# --- Inverse FFT to time domain ---
left_channel = np.fft.irfft(fft_left) * background_volume
right_channel = np.fft.irfft(fft_right) * background_volume

# Optional: boost the bandpass portion for clarity
left_channel += np.fft.irfft(fft_band) * (bandpass_volume - background_volume)
right_channel += np.fft.irfft(fft_band_shifted) * (bandpass_volume - background_volume)

# --- Combine into stereo ---
stimulus_huggins = np.column_stack((left_channel, right_channel))

# --- Create test sounds ---
# 1. Broadband noise (identical to both ears)
stimulus_noise = np.column_stack((noise, noise))

# 2. Pure tone at same frequency
t = np.arange(n_samples) / fs
tone = np.sin(2 * np.pi * bandCentre * t)
stimulus_tone = np.column_stack((tone, tone))

# --- Test sequence ---
print("Test 1: Broadband noise (control)...")
sd.play(stimulus_noise / np.max(np.abs(stimulus_noise)), fs)
sd.wait()

print("Test 2: Pure tone at {:.1f} Hz...".format(bandCentre))
sd.play(stimulus_tone / np.max(np.abs(stimulus_tone)), fs)
sd.wait()

print("Test 3: Huggins pitch (dichotic)...")
sd.play(stimulus_huggins / np.max(np.abs(stimulus_huggins)), fs)
sd.wait()

print("Done.")
