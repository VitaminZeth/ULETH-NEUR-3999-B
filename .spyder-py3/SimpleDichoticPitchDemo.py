#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 12:19:25 2025

@author: Seth Villamil & ChatGPT
"""
import numpy as np
# from scipy.signal.windows import gaussian
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
bin_width_hz = fs / n_samples     # Hz per bin
bandwidth_bins = bandwidth_hz / bin_width_hz

# --- Convert to Gaussian std in bins ---
std = bandwidth_bins / 2.355

# --- Create aligned Gaussian windows in the frequency (bin) domain ---

M = len(fft_noise)                # number of RFFT bins (0..N/2)

def freq_to_bin(f_hz: float) -> int:
    """Clamp frequency to valid range and map to nearest RFFT bin index."""
    f_hz = max(0.0, min(f_hz, fs / 2.0))
    return int(round(f_hz / bin_width_hz))

# std is already in bins (bandwidth_bins / 2.355)
sigma_bins = max(std, 1e-9)

def gaussian_at_bin(center_bin: int, sigma_bins: float, num_bins: int) -> np.ndarray:
    """Unit-peak Gaussian centered at center_bin, defined over RFFT bins 0..num_bins-1."""
    n = np.arange(num_bins)
    return np.exp(-0.5 * ((n - center_bin) / sigma_bins) ** 2)

# Centers in bin indices
cbin1 = freq_to_bin(bandCentre1)
cbin2 = freq_to_bin(bandCentre2)

# Band-pass windows centered exactly at bandCentre* (unit peak = 1.0)
window1 = gaussian_at_bin(cbin1, sigma_bins, M)
window2 = gaussian_at_bin(cbin2, sigma_bins, M)

# Complementary notches that null the band-pass region
complementaryWindow1 = 1.0 - window1
complementaryWindow2 = 1.0 - window2

# Optional: sanity check
print(f"BandCentre1: {bandCentre1} Hz -> bin {cbin1}, actual freq {cbin1*bin_width_hz:.2f} Hz")
print(f"BandCentre2: {bandCentre2} Hz -> bin {cbin2}, actual freq {cbin2*bin_width_hz:.2f} Hz")

from datetime import datetime

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# --- Plot folders ---
plots_dir = outdir / "plots"
windows_dir = plots_dir / "windows"
spectra_dir = plots_dir / "spectra"
for d in (plots_dir, windows_dir, spectra_dir):
    d.mkdir(parents=True, exist_ok=True)
print(f"Plots will be saved to: {plots_dir}")

# =========================
# Window visualizations (aligned) — high-res, full/zoom/log
# =========================
# Scale windows to [0,1] for overlays and consistent y-axis
w_left_bp  = window1 / (np.max(window1) if np.max(window1) > 0 else 1.0)
w_left_nt  = complementaryWindow1 / (np.max(complementaryWindow1) if np.max(complementaryWindow1) > 0 else 1.0)
w_right_bp = window2 / (np.max(window2) if np.max(window2) > 0 else 1.0)
w_right_nt = complementaryWindow2 / (np.max(complementaryWindow2) if np.max(complementaryWindow2) > 0 else 1.0)

# --- Overview: both ears (linear x) ---
plt.figure(figsize=(12, 6))
plt.title("Filter Windows (Aligned) — Both Ears")
plt.plot(freqs, w_left_bp,  label="Left Band-pass (scaled)")
plt.plot(freqs, w_left_nt,  label="Left Notch (scaled)", linestyle="--")
plt.plot(freqs, w_right_bp, label="Right Band-pass (scaled)")
plt.plot(freqs, w_right_nt, label="Right Notch (scaled)", linestyle="--")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Gain (scaled)")
plt.legend(loc="best")
plt.xlim(0, fs/2)
plt.tight_layout()
plt.savefig(windows_dir / "02a_windows_overview_linear.png", dpi=300)
plt.close()

# --- Overview: both ears (log x) ---
plt.figure(figsize=(12, 6))
plt.title("Filter Windows (Aligned, Log-x) — Both Ears")
plt.semilogx(freqs, w_left_bp,  label="Left Band-pass (scaled)")
plt.semilogx(freqs, w_left_nt,  label="Left Notch (scaled)", linestyle="--")
plt.semilogx(freqs, w_right_bp, label="Right Band-pass (scaled)")
plt.semilogx(freqs, w_right_nt, label="Right Notch (scaled)", linestyle="--")
plt.xlabel("Frequency (Hz, log scale)")
plt.ylabel("Gain (scaled)")
plt.legend(loc="best")
plt.xlim(20, fs/2)
plt.tight_layout()
plt.savefig(windows_dir / "02b_windows_overview_log.png", dpi=300)
plt.close()

# --- Left ear: full (linear x) ---
plt.figure(figsize=(12, 6))
plt.title("Left Ear — Window Shapes (Aligned)")
plt.plot(freqs, w_left_bp, label="Left Band-pass (scaled)")
plt.plot(freqs, w_left_nt, label="Left Notch (scaled)", linestyle="--")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Gain (scaled)")
plt.legend(loc="best")
plt.xlim(0, fs/2)
plt.tight_layout()
plt.savefig(windows_dir / "02c_left_windows_linear.png", dpi=300)
plt.close()

# --- Left ear: zoom around bandCentre1 ± 3×bandwidth_hz (linear x) ---
plt.figure(figsize=(12, 6))
plt.title("Left Ear — Window Zoom Around Band Centre")
plt.plot(freqs, w_left_bp, label="Left Band-pass (scaled)")
plt.plot(freqs, w_left_nt, label="Left Notch (scaled)", linestyle="--")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Gain (scaled)")
plt.legend(loc="best")
plt.xlim(bandCentre1 - 3*bandwidth_hz, bandCentre1 + 3*bandwidth_hz)
plt.tight_layout()
plt.savefig(windows_dir / "02d_left_windows_zoom.png", dpi=300)
plt.close()

# --- Left ear: full (log x) ---
plt.figure(figsize=(12, 6))
plt.title("Left Ear — Window Shapes (Aligned, Log-x)")
plt.semilogx(freqs, w_left_bp, label="Left Band-pass (scaled)")
plt.semilogx(freqs, w_left_nt, label="Left Notch (scaled)", linestyle="--")
plt.xlabel("Frequency (Hz, log scale)")
plt.ylabel("Gain (scaled)")
plt.legend(loc="best")
plt.xlim(20, fs/2)
plt.tight_layout()
plt.savefig(windows_dir / "02e_left_windows_log.png", dpi=300)
plt.close()

# --- Right ear: full (linear x) ---
plt.figure(figsize=(12, 6))
plt.title("Right Ear — Window Shapes (Aligned)")
plt.plot(freqs, w_right_bp, label="Right Band-pass (scaled)")
plt.plot(freqs, w_right_nt, label="Right Notch (scaled)", linestyle="--")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Gain (scaled)")
plt.legend(loc="best")
plt.xlim(0, fs/2)
plt.tight_layout()
plt.savefig(windows_dir / "02f_right_windows_linear.png", dpi=300)
plt.close()

# --- Right ear: zoom around bandCentre2 ± 3×bandwidth_hz (linear x) ---
plt.figure(figsize=(12, 6))
plt.title("Right Ear — Window Zoom Around Band Centre")
plt.plot(freqs, w_right_bp, label="Right Band-pass (scaled)")
plt.plot(freqs, w_right_nt, label="Right Notch (scaled)", linestyle="--")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Gain (scaled)")
plt.legend(loc="best")
plt.xlim(bandCentre2 - 3*bandwidth_hz, bandCentre2 + 3*bandwidth_hz)
plt.tight_layout()
plt.savefig(windows_dir / "02g_right_windows_zoom.png", dpi=300)
plt.close()

# --- Right ear: full (log x) ---
plt.figure(figsize=(12, 6))
plt.title("Right Ear — Window Shapes (Aligned, Log-x)")
plt.semilogx(freqs, w_right_bp, label="Right Band-pass (scaled)")
plt.semilogx(freqs, w_right_nt, label="Right Notch (scaled)", linestyle="--")
plt.xlabel("Frequency (Hz, log scale)")
plt.ylabel("Gain (scaled)")
plt.legend(loc="best")
plt.xlim(20, fs/2)
plt.tight_layout()
plt.savefig(windows_dir / "02h_right_windows_log.png", dpi=300)
plt.close()

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
    f"cbin1: {cbin1} (realized center {cbin1*bin_width_hz:.2f} Hz)",
    f"cbin2: {cbin2} (realized center {cbin2*bin_width_hz:.2f} Hz)",
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

# =========================
# FFT export + visualization
# =========================
import csv

# Make dedicated subfolder for FFT CSVs
fft_csv_dir = outdir / "fft_csv"
fft_csv_dir.mkdir(parents=True, exist_ok=True)

# ---- Helper for exporting FFTs ----
def export_fft_csv(signal, fs, filename):
    """Export FFT of a 1D signal to CSV with columns: Frequency, Magnitude_dBFS, Phase."""
    fft_vals = np.fft.rfft(signal)
    freqs_local = np.fft.rfftfreq(len(signal), 1/fs)
    mag = np.abs(fft_vals)
    mag_ref = np.max(mag) if np.max(mag) > 0 else 1.0
    mag_dbfs = 20.0 * np.log10(np.clip(mag / mag_ref, 1e-12, None))
    phase = np.angle(fft_vals)

    path = fft_csv_dir / filename
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Frequency_Hz", "Magnitude_dBFS", "Phase_rad"])
        for fi, mi, pi in zip(freqs_local, mag_dbfs, phase):
            w.writerow([fi, mi, pi])
    print(f"Saved FFT CSV to: {path}")

# ---- 1) Export FFTs to CSV (noise + mono mixes) ----
export_fft_csv(noise, fs, "01_noise_fft.csv")
export_fft_csv(mixedl, fs, "02_left_mix_fft.csv")
export_fft_csv(mixedr, fs, "03_right_mix_fft.csv")

# ---- 2) Export Stereo FFT CSV (left & right together) ----
fft_left = np.fft.rfft(mixedl)
fft_right = np.fft.rfft(mixedr)
freqs_lr = np.fft.rfftfreq(len(mixedl), 1/fs)

magL = np.abs(fft_left); magR = np.abs(fft_right)
mag_ref = max(np.max(magL), np.max(magR), 1.0)  # use max of both channels
magL_dbfs = 20.0 * np.log10(np.clip(magL / mag_ref, 1e-12, None))
magR_dbfs = 20.0 * np.log10(np.clip(magR / mag_ref, 1e-12, None))
phaseL = np.angle(fft_left); phaseR = np.angle(fft_right)

stereo_csv_path = fft_csv_dir / "04_stereo_mix_fft.csv"
with open(stereo_csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["Frequency_Hz",
                "Left_Mag_dBFS", "Left_Phase_rad",
                "Right_Mag_dBFS", "Right_Phase_rad"])
    for fi, ml, pl, mr, pr in zip(freqs_lr, magL_dbfs, phaseL, magR_dbfs, phaseR):
        w.writerow([fi, ml, pl, mr, pr])

print(f"Saved stereo FFT CSV to: {stereo_csv_path}")

# =========================
# Hi-res spectral visualizations (self-contained)
# =========================
# Compute spectra and normalized magnitudes
fft_left = np.fft.rfft(mixedl)
fft_right = np.fft.rfft(mixedr)
magL = np.abs(fft_left)
magR = np.abs(fft_right)
left_mag_norm = magL / (np.max(magL) if np.max(magL) > 0 else 1.0)
right_mag_norm = magR / (np.max(magR) if np.max(magR) > 0 else 1.0)

# Recompute scaled windows (if needed earlier, these lines are harmless)
w_left_bp = window1 / (np.max(window1) if np.max(window1) > 0 else 1.0)
w_left_nt = complementaryWindow1 / (np.max(complementaryWindow1) if np.max(complementaryWindow1) > 0 else 1.0)
w_right_bp = window2 / (np.max(window2) if np.max(window2) > 0 else 1.0)
w_right_nt = complementaryWindow2 / (np.max(complementaryWindow2) if np.max(complementaryWindow2) > 0 else 1.0)

# ---- Filters overview (both ears), hi-res ----
plt.figure(figsize=(12, 6))
plt.title("Filter Windows (Band-pass and Complementary Notch) - Both Ears")
plt.plot(freqs, w_left_bp, label="Left: Band-pass (scaled)")
plt.plot(freqs, w_left_nt, label="Left: Notch (scaled)", linestyle="--")
plt.plot(freqs, w_right_bp, label="Right: Band-pass (scaled)")
plt.plot(freqs, w_right_nt, label="Right: Notch (scaled)", linestyle="--")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Gain (scaled)")
plt.legend(loc="best")
plt.xlim(0, fs/2)
plt.tight_layout()
plt.savefig(spectra_dir / "02_filters_overview_left_right.png", dpi=300)
plt.close()

# ---- Left ear: full spectrum (linear x) ----
plt.figure(figsize=(12, 6))
plt.title("Left Ear Spectrum with Filter Windows")
plt.plot(freqs, left_mag_norm, label="Left Spectrum (|FFT|, normalized)")
plt.plot(freqs, w_left_bp, label="Left Band-pass (scaled)", linestyle="--")
plt.plot(freqs, w_left_nt, label="Left Notch (scaled)", linestyle=":")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Normalized magnitude")
plt.legend(loc="best")
plt.xlim(0, fs/2)
plt.tight_layout()
plt.savefig(spectra_dir / "03_left_spectrum_with_filters.png", dpi=300)
plt.close()

# ---- Left ear: zoom around band centre ±3×bandwidth ----
plt.figure(figsize=(12, 6))
plt.title("Left Ear Spectrum (Zoomed Near Band Centre)")
plt.plot(freqs, left_mag_norm, label="Left Spectrum (|FFT|, normalized)")
plt.plot(freqs, w_left_bp, label="Left Band-pass (scaled)", linestyle="--")
plt.plot(freqs, w_left_nt, label="Left Notch (scaled)", linestyle=":")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Normalized magnitude")
plt.legend(loc="best")
plt.xlim(bandCentre1 - 3*bandwidth_hz, bandCentre1 + 3*bandwidth_hz)
plt.tight_layout()
plt.savefig(spectra_dir / "03_left_spectrum_zoom.png", dpi=300)
plt.close()

# ---- Left ear: log-scale frequency ----
plt.figure(figsize=(12, 6))
plt.title("Left Ear Spectrum (Log-Scale Frequency Axis)")
plt.semilogx(freqs, left_mag_norm, label="Left Spectrum (|FFT|, normalized)")
plt.semilogx(freqs, w_left_bp, label="Left Band-pass (scaled)", linestyle="--")
plt.semilogx(freqs, w_left_nt, label="Left Notch (scaled)", linestyle=":")
plt.xlabel("Frequency (Hz, log scale)")
plt.ylabel("Normalized magnitude")
plt.legend(loc="best")
plt.xlim(20, fs/2)
plt.tight_layout()
plt.savefig(spectra_dir / "03_left_spectrum_log.png", dpi=300)
plt.close()

# ---- Right ear: full spectrum (linear x) ----
plt.figure(figsize=(12, 6))
plt.title("Right Ear Spectrum with Filter Windows")
plt.plot(freqs, right_mag_norm, label="Right Spectrum (|FFT|, normalized)")
plt.plot(freqs, w_right_bp, label="Right Band-pass (scaled)", linestyle="--")
plt.plot(freqs, w_right_nt, label="Right Notch (scaled)", linestyle=":")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Normalized magnitude")
plt.legend(loc="best")
plt.xlim(0, fs/2)
plt.tight_layout()
plt.savefig(spectra_dir / "04_right_spectrum_with_filters.png", dpi=300)
plt.close()

# ---- Right ear: zoom around band centre ±3×bandwidth ----
plt.figure(figsize=(12, 6))
plt.title("Right Ear Spectrum (Zoomed Near Band Centre)")
plt.plot(freqs, right_mag_norm, label="Right Spectrum (|FFT|, normalized)")
plt.plot(freqs, w_right_bp, label="Right Band-pass (scaled)", linestyle="--")
plt.plot(freqs, w_right_nt, label="Right Notch (scaled)", linestyle=":")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Normalized magnitude")
plt.legend(loc="best")
plt.xlim(bandCentre2 - 3*bandwidth_hz, bandCentre2 + 3*bandwidth_hz)
plt.tight_layout()
plt.savefig(spectra_dir / "04_right_spectrum_zoom.png", dpi=300)
plt.close()

# ---- Right ear: log-scale frequency ----
plt.figure(figsize=(12, 6))
plt.title("Right Ear Spectrum (Log-Scale Frequency Axis)")
plt.semilogx(freqs, right_mag_norm, label="Right Spectrum (|FFT|, normalized)")
plt.semilogx(freqs, w_right_bp, label="Right Band-pass (scaled)", linestyle="--")
plt.semilogx(freqs, w_right_nt, label="Right Notch (scaled)", linestyle=":")
plt.xlabel("Frequency (Hz, log scale)")
plt.ylabel("Normalized magnitude")
plt.legend(loc="best")
plt.xlim(20, fs/2)
plt.tight_layout()
plt.savefig(spectra_dir / "04_right_spectrum_log.png", dpi=300)
plt.close()

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

print("Playing shifted notch-filtered noise...")
sd.play(advanced_notched2 / (np.max(np.abs(advanced_notched2)) if np.max(np.abs(advanced_notched2)) > 0 else 1.0), fs)
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

# Print out the audio files at each stage after 
# Ask Georg Boenn About looking into the code
# Make three melodies for the tests
# play some control trials or practice trials
# Sham trials with no melodies
# Stack of puretones (Fund and its harmonics), pull out the harmonic in space and see if i can hear it
# 