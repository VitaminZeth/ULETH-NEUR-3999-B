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
from pathlib import Path
from scipy.io.wavfile import write as wavwrite

# Tiny Helper
def to_int16(x: np.ndarray) -> np.ndarray:
    """Normalize to full scale and convert to 16-bit PCM."""
    peak = np.max(np.abs(x)) if np.size(x) else 0.0
    if peak == 0 or not np.isfinite(peak):
        peak = 1.0
    return (x / peak * 32767.0).astype(np.int16)

# --- Parameters ---
fs = 44100          # Sampling rate
# sd.default.samplerate = fs # Set default sd Sampling rate
duration = 1.0      # seconds
n_samples = int(fs * duration)

# --- Delay parameter in milliseconds ---
delay_ms = 0.6  # e.g. 2 milliseconds

# Equation to convert ms to samples
delay_samples = int(round((delay_ms / 1000.0) * fs))

# --- Generate broadband noise ---
noise = np.random.normal(0, 1, n_samples)

# --- FFT of the noise ---
fft_noise = np.fft.rfft(noise)
freqs = np.fft.rfftfreq(n_samples, 1/fs)

# --- Assigning the centre frequencies ---
bandCentre1 = 500  # Hz
bandCentre2 = 500  # Hz

# --- Desired filter bandwidth: 1/20th of center frequency ---
bandwidth_hz = bandCentre1 / 20.0
bin_width_hz = fs / n_samples
bandwidth_bins = bandwidth_hz / bin_width_hz

# --- Convert to Gaussian std in bins ---
std = bandwidth_bins / 2.355

# --- Create Gaussian window ---
M = len(fft_noise)
window = gaussian(M, std)
mid = len(freqs) // 2

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

# --- Volume parameter for the bandpass components ---
bandpass_volume = 1.0  # 1.0 = original level, <1 = quieter, >1 = louder
background_volume = 1.0   # controls loudness of the notched (background) noise

# --- Apply Gaussian bandpass
fft_bandpassed1 = fft_noise * window1 # Freq Domain
bandpassed1 = np.fft.irfft(fft_bandpassed1) * bandpass_volume #plot the points

# --- Apply a shifted Gaussian bandpass to a copy
fft_bandpassed2 = fft_noise * window2 # Freq Domain
bandpassed2 = np.fft.irfft(fft_bandpassed2) * bandpass_volume #plot the points

# --- Apply a sample delay to the shifted Gaussian bandpassed copy

# # Define the delay in samples
# delay_samples = 10  # Adjust as needed

# # Create a zero-padded array for the delay
# delayed_bandpassed2 = np.pad(bandpassed2, (delay_samples, 0), 'constant')
# ValueError: operands could not be broadcast together with shapes (88210,) (88200,) 

# Shift by x samples to the right (delay) 
delayed_bandpassed2 = np.roll(bandpassed2, delay_samples)
print(f"Delayed signal (delay integer shift): {delayed_bandpassed2}")

# # Shift by x samples to the left (advance)
# advanced_bandpassed2 = np.roll(signal, -2)
# print(f"Advanced signal (advance integer shift): {advanced_signal}")

# --- Design complementary notch (1 - Gaussian) ---
fft_notched1 = fft_noise * complementaryWindow1 # Freq Domain
notched1 = np.fft.irfft(fft_notched1) * background_volume #plot the points

# --- Design complementary shifted notch (1 - Gaussian) ---
fft_notched2 = fft_noise * complementaryWindow2 # Freq Domain
notched2 = np.fft.irfft(fft_notched2) * background_volume #plot the points

# Shift by x samples to the right (delay) 
# delayed_signal = np.roll(signal, -2)
# print(f"delayed signal (advance integer shift): {delayed_signal}")

# Shift by x samples to the left (advance) 
advanced_notched2 = np.roll(notched2, -delay_samples)
print(f"Advanced signal (delay integer shift): {advanced_notched2}")

# --- Blend the files
mixedl = (bandpassed1 + notched1)
mixedr = (delayed_bandpassed2 + advanced_notched2)
                                            
# --- Combine into a stereo array (two columns)                                            
mixed = np.column_stack((mixedl, mixedr))

from datetime import datetime

# === Exports: make folder and write WAVs (44100 Hz, 16-bit) ===
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
outdir = Path(__file__).resolve().parent / "[to_analyze]" / timestamp
outdir.mkdir(parents=True, exist_ok=True)

print(f"\nRendered WAVs will be saved to: {outdir}\n")

# ---- Write a parameters snapshot for this run ----
# Captures key values from setup (lines ~23–57) and volumes (lines ~76–78)
params_lines = [
    f"Run timestamp: {timestamp}",
    f"Output folder: {outdir}",
    "",
    "[Signal & Timing]",
    f"fs (Hz): {fs}",
    f"duration (s): {duration}",
    f"n_samples: {n_samples}",
    f"delay_ms: {delay_ms}",
    f"delay_samples: {delay_samples}",
    "",
    "[Frequency & Windows]",
    f"bandCentre1 (Hz): {bandCentre1}",
    f"bandCentre2 (Hz): {bandCentre2}",
    f"bandwidth_hz (center/20): {bandwidth_hz}",
    f"bin_width_hz: {bin_width_hz}",
    f"bandwidth_bins: {bandwidth_bins}",
    f"gaussian_std_bins (std): {std}",
    f"fft_length_M: {M}",
    f"mid_index: {mid}",
    "",
    "[Levels]",
    f"bandpass_volume: {bandpass_volume}",
    f"background_volume: {background_volume}",
    "",
    "[Notes]",
    "This file logs the parameter values used to render the WAVs in this run."
]

with open(outdir / "00_run_parameters.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(params_lines))

print(f"Wrote parameters snapshot to: {outdir / '00_run_parameters.txt'}")

# Originals and individual filters
wavwrite(outdir / "00_original_noise.wav", fs, to_int16(noise))
wavwrite(outdir / "10_bandpass_ch1.wav", fs, to_int16(bandpassed1))
wavwrite(outdir / "11_bandpass_ch2_delayed.wav", fs, to_int16(delayed_bandpassed2))
wavwrite(outdir / "20_notch_ch1.wav", fs, to_int16(notched1))
wavwrite(outdir / "21_notch_ch2_advanced.wav", fs, to_int16(advanced_notched2))

# Per-channel mixes
wavwrite(outdir / "30_mixed_left.wav", fs, to_int16(mixedl))
wavwrite(outdir / "31_mixed_right.wav", fs, to_int16(mixedr))

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
sd.play(notched2 / np.max(np.abs(advanced_notched2)), fs)
sd.wait()

# --- Play the notched and band-passed together (mono)
print("Playing Mixed Left Channel...")
sd.play(mixedl / np.max(np.abs(mixedl)), fs)
sd.wait()

# --- Play the notched and band-passed together (mono)
print("Playing Mixed Right Channel...")
sd.play(mixedr / np.max(np.abs(mixedr)), fs)
sd.wait()

print("Playing Stimuli 1 Time...")

print("Playing Mixed L + R... 1")
played_stereo = mixed / (np.max(np.abs(mixed)) if np.max(np.abs(mixed)) > 0 else 1.0)
sd.play(played_stereo, fs)
sd.wait()

# Save exactly what was played to your ears
wavwrite(outdir / "40_played_stereo_final.wav", fs, to_int16(played_stereo))
print(f"Saved final played stereo render to: {outdir / '40_played_stereo_final.wav'}")

# # --- Play the notched and band-passedtogether (stereo)
# print("Playing Mixed L + R... 2")
# sd.play(mixed / np.max(np.abs(mixed)), fs)
# sd.wait()

# # --- Play the notched and band-passedtogether (stereo)
# print("Playing Mixed L + R... 3")
# sd.play(mixed / np.max(np.abs(mixed)), fs)
# sd.wait()

# # --- Play the notched and band-passedtogether (stereo)
# print("Playing Mixed L + R... 4")
# sd.play(mixed / np.max(np.abs(mixed)), fs)
# sd.wait()

# NOTE TO SELF:
#   Everythign worked out for the copy, but I need to fix it so that the playback of all the individual files works
#   as well as the notch staying the same, and only moving the band-pass