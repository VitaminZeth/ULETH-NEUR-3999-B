#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Window-Based Dichotic Pitch (Pentatonic Melodies) — FULL SCRIPT with Original-Style Plots
---------------------------------------------------------------------------------------
Generates FIVE stereo WAV files.
Each file contains two 8-note pentatonic melodies (Left/Right).
Each NOTE is generated EXACTLY using your original window-based method:
    noise -> RFFT -> multiply by Gaussian window & (1-window) -> IRFFT
    right bandpass delayed, right notch advanced
    mix left = bp1+nt1, right = delayed(bp2)+advanced(nt2)

Plots:
For EACH note, saves the same plot types you created originally:
Windows (full/zoom/log) and Spectra (full/zoom/log) into:
    [to_analyze]/timestamp/plots/windows/
    [to_analyze]/timestamp/plots/spectra/

Requires:
    pip install numpy scipy matplotlib
Optional:
    pip install sounddevice
"""

import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.io.wavfile import write as wavwrite

import matplotlib.pyplot as plt

# Optional playback
try:
    import sounddevice as sd
    HAVE_SD = True
except Exception:
    HAVE_SD = False


# -----------------------
# Tiny Helper (kept from your original)
# -----------------------
def to_int16(x: np.ndarray) -> np.ndarray:
    """Normalize to full scale and convert to 16-bit PCM."""
    peak = np.max(np.abs(x)) if np.size(x) else 0.0
    if peak == 0 or not np.isfinite(peak):
        peak = 1.0
    return (x / peak * 32767.0).astype(np.int16)


def _safe_norm(x: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(x)) if np.size(x) else 0.0
    return x / (peak if peak > 0 else 1.0)


# -----------------------
# Pentatonic degrees (semitones relative to base)
# -----------------------
PENTA_MAJOR = [0, 2, 4, 7, 9]   # major pentatonic
PENTA_MINOR = [0, 3, 5, 7, 10]  # minor pentatonic


def build_pentatonic_melody(degrees: list[int], idx_seq: list[int]) -> list[int]:
    """Convert degree indices (0..4) into semitone offsets (length must be 8)."""
    return [int(degrees[i % len(degrees)]) for i in idx_seq]


def semitones_to_hz(base_hz: float, semitones: float) -> float:
    return float(base_hz) * (2.0 ** (float(semitones) / 12.0))


def apply_fade(note: np.ndarray, fs: int, fade_ms: float = 10.0) -> np.ndarray:
    """Small fade to avoid clicks at note boundaries (doesn't change your window method)."""
    n = len(note)
    fade_n = int(round((fade_ms / 1000.0) * fs))
    fade_n = max(1, min(fade_n, n // 2))
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, fade_n))
    out = note.copy()
    out[:fade_n] *= ramp
    out[-fade_n:] *= ramp[::-1]
    return out


# -----------------------
# EXACT window-based method helpers (bin-domain Gaussian)
# -----------------------
def freq_to_bin(f_hz: float, bin_width_hz: float, fs: int) -> int:
    f_hz = max(0.0, min(float(f_hz), fs / 2.0))
    return int(round(f_hz / bin_width_hz))


def gaussian_at_bin(center_bin: int, sigma_bins: float, num_bins: int) -> np.ndarray:
    n = np.arange(num_bins)
    sigma_bins = max(float(sigma_bins), 1e-9)
    return np.exp(-0.5 * ((n - center_bin) / sigma_bins) ** 2)


# -----------------------
# Plotting (copied in STYLE from your original)
# -----------------------
def save_window_plots(
    *,
    windows_dir: Path,
    freqs: np.ndarray,
    window1: np.ndarray,
    window2: np.ndarray,
    complementaryWindow1: np.ndarray,
    complementaryWindow2: np.ndarray,
    fs: int,
    bandCentre1_hz: float,
    bandCentre2_hz: float,
    bandwidth1_hz: float,
    bandwidth2_hz: float,
    prefix: str
) -> None:
    """Save the same window plot set as your original script (full/zoom/log)."""

    # Scale to [0,1] for consistent overlays (exactly like you did)
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
    plt.savefig(windows_dir / f"{prefix}_02a_windows_overview_linear.png", dpi=300)
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
    plt.savefig(windows_dir / f"{prefix}_02b_windows_overview_log.png", dpi=300)
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
    plt.savefig(windows_dir / f"{prefix}_02c_left_windows_linear.png", dpi=300)
    plt.close()

    # --- Left ear: zoom around bandCentre1 ± 3×bandwidth (linear x) ---
    plt.figure(figsize=(12, 6))
    plt.title("Left Ear — Window Zoom Around Band Centre")
    plt.plot(freqs, w_left_bp, label="Left Band-pass (scaled)")
    plt.plot(freqs, w_left_nt, label="Left Notch (scaled)", linestyle="--")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Gain (scaled)")
    plt.legend(loc="best")
    plt.xlim(bandCentre1_hz - 3*bandwidth1_hz, bandCentre1_hz + 3*bandwidth1_hz)
    plt.tight_layout()
    plt.savefig(windows_dir / f"{prefix}_02d_left_windows_zoom.png", dpi=300)
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
    plt.savefig(windows_dir / f"{prefix}_02e_left_windows_log.png", dpi=300)
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
    plt.savefig(windows_dir / f"{prefix}_02f_right_windows_linear.png", dpi=300)
    plt.close()

    # --- Right ear: zoom around bandCentre2 ± 3×bandwidth (linear x) ---
    plt.figure(figsize=(12, 6))
    plt.title("Right Ear — Window Zoom Around Band Centre")
    plt.plot(freqs, w_right_bp, label="Right Band-pass (scaled)")
    plt.plot(freqs, w_right_nt, label="Right Notch (scaled)", linestyle="--")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Gain (scaled)")
    plt.legend(loc="best")
    plt.xlim(bandCentre2_hz - 3*bandwidth2_hz, bandCentre2_hz + 3*bandwidth2_hz)
    plt.tight_layout()
    plt.savefig(windows_dir / f"{prefix}_02g_right_windows_zoom.png", dpi=300)
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
    plt.savefig(windows_dir / f"{prefix}_02h_right_windows_log.png", dpi=300)
    plt.close()


def save_spectra_plots(
    *,
    spectra_dir: Path,
    freqs: np.ndarray,
    mixedl: np.ndarray,
    mixedr: np.ndarray,
    window1: np.ndarray,
    window2: np.ndarray,
    complementaryWindow1: np.ndarray,
    complementaryWindow2: np.ndarray,
    fs: int,
    bandCentre1_hz: float,
    bandCentre2_hz: float,
    bandwidth1_hz: float,
    bandwidth2_hz: float,
    prefix: str
) -> None:
    """Save the same spectra plot set as your original script (full/zoom/log)."""

    # Spectra
    fft_left = np.fft.rfft(mixedl)
    fft_right = np.fft.rfft(mixedr)
    magL = np.abs(fft_left)
    magR = np.abs(fft_right)
    left_mag_norm = magL / (np.max(magL) if np.max(magL) > 0 else 1.0)
    right_mag_norm = magR / (np.max(magR) if np.max(magR) > 0 else 1.0)

    # Scaled windows (same as original)
    w_left_bp = window1 / (np.max(window1) if np.max(window1) > 0 else 1.0)
    w_left_nt = complementaryWindow1 / (np.max(complementaryWindow1) if np.max(complementaryWindow1) > 0 else 1.0)
    w_right_bp = window2 / (np.max(window2) if np.max(window2) > 0 else 1.0)
    w_right_nt = complementaryWindow2 / (np.max(complementaryWindow2) if np.max(complementaryWindow2) > 0 else 1.0)

    # ---- Filters overview (both ears) ----
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
    plt.savefig(spectra_dir / f"{prefix}_02_filters_overview_left_right.png", dpi=300)
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
    plt.savefig(spectra_dir / f"{prefix}_03_left_spectrum_with_filters.png", dpi=300)
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
    plt.xlim(bandCentre1_hz - 3*bandwidth1_hz, bandCentre1_hz + 3*bandwidth1_hz)
    plt.tight_layout()
    plt.savefig(spectra_dir / f"{prefix}_03_left_spectrum_zoom.png", dpi=300)
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
    plt.savefig(spectra_dir / f"{prefix}_03_left_spectrum_log.png", dpi=300)
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
    plt.savefig(spectra_dir / f"{prefix}_04_right_spectrum_with_filters.png", dpi=300)
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
    plt.xlim(bandCentre2_hz - 3*bandwidth2_hz, bandCentre2_hz + 3*bandwidth2_hz)
    plt.tight_layout()
    plt.savefig(spectra_dir / f"{prefix}_04_right_spectrum_zoom.png", dpi=300)
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
    plt.savefig(spectra_dir / f"{prefix}_04_right_spectrum_log.png", dpi=300)
    plt.close()


# -----------------------
# One NOTE using your exact pipeline + plots
# -----------------------
def make_one_note_window_based(
    *,
    fs: int,
    note_n: int,
    delay_samples: int,
    bandCentre1_hz: float,
    bandCentre2_hz: float,
    bandpass_volume: float,
    background_volume: float,
    rng: np.random.Generator,
    fade_ms: float,
    windows_dir: Path,
    spectra_dir: Path,
    plot_prefix: str
) -> np.ndarray:
    # Broadband noise (fresh per note)
    noise = rng.normal(0, 1, note_n)

    # FFT of noise
    fft_noise = np.fft.rfft(noise)
    M = len(fft_noise)

    # Frequency axis for THIS note (matches your original method)
    freqs = np.fft.rfftfreq(note_n, 1/fs)

    # Bin width for THIS note length
    bin_width_hz = fs / note_n

    # Your exact bandwidth rule: center/20
    bandwidth1_hz = bandCentre1_hz / 20.0
    bandwidth2_hz = bandCentre2_hz / 20.0

    bandwidth1_bins = bandwidth1_hz / bin_width_hz
    bandwidth2_bins = bandwidth2_hz / bin_width_hz

    # Your exact std conversion: std = bandwidth_bins / 2.355
    std1 = bandwidth1_bins / 2.355
    std2 = bandwidth2_bins / 2.355

    sigma_bins1 = max(std1, 1e-9)
    sigma_bins2 = max(std2, 1e-9)

    # Centers in bin indices
    cbin1 = freq_to_bin(bandCentre1_hz, bin_width_hz, fs)
    cbin2 = freq_to_bin(bandCentre2_hz, bin_width_hz, fs)

    # Gaussian windows
    window1 = gaussian_at_bin(cbin1, sigma_bins1, M)
    window2 = gaussian_at_bin(cbin2, sigma_bins2, M)

    # Complementary notches
    complementaryWindow1 = 1.0 - window1
    complementaryWindow2 = 1.0 - window2

    # ===== Window plots (exact set like your original) =====
    save_window_plots(
        windows_dir=windows_dir,
        freqs=freqs,
        window1=window1,
        window2=window2,
        complementaryWindow1=complementaryWindow1,
        complementaryWindow2=complementaryWindow2,
        fs=fs,
        bandCentre1_hz=bandCentre1_hz,
        bandCentre2_hz=bandCentre2_hz,
        bandwidth1_hz=bandwidth1_hz,
        bandwidth2_hz=bandwidth2_hz,
        prefix=plot_prefix
    )

    # Apply Gaussian bandpass
    fft_bandpassed1 = fft_noise * window1
    bandpassed1 = np.fft.irfft(fft_bandpassed1) * bandpass_volume

    fft_bandpassed2 = fft_noise * window2
    bandpassed2 = np.fft.irfft(fft_bandpassed2) * bandpass_volume

    # Delay right bandpass
    delayed_bandpassed2 = np.roll(bandpassed2, delay_samples)

    # Apply complementary notch
    fft_notched1 = fft_noise * complementaryWindow1
    notched1 = np.fft.irfft(fft_notched1) * background_volume

    fft_notched2 = fft_noise * complementaryWindow2
    notched2 = np.fft.irfft(fft_notched2) * background_volume

    # Advance right notch
    advanced_notched2 = np.roll(notched2, -delay_samples)

    # Blend (same mix rule)
    mixedl = (bandpassed1 + notched1)
    mixedr = (delayed_bandpassed2 + advanced_notched2)

    # Fade note edges
    mixedl = apply_fade(mixedl, fs, fade_ms=fade_ms)
    mixedr = apply_fade(mixedr, fs, fade_ms=fade_ms)

    # ===== Spectra plots (exact set like your original) =====
    save_spectra_plots(
        spectra_dir=spectra_dir,
        freqs=freqs,
        mixedl=mixedl,
        mixedr=mixedr,
        window1=window1,
        window2=window2,
        complementaryWindow1=complementaryWindow1,
        complementaryWindow2=complementaryWindow2,
        fs=fs,
        bandCentre1_hz=bandCentre1_hz,
        bandCentre2_hz=bandCentre2_hz,
        bandwidth1_hz=bandwidth1_hz,
        bandwidth2_hz=bandwidth2_hz,
        prefix=plot_prefix
    )

    return np.column_stack((mixedl, mixedr))


def render_one_file_two_melodies(
    *,
    fs: int,
    note_dur_s: float,
    delay_ms: float,
    base_center_hz: float,
    melody_L_semitones: list[int],
    melody_R_semitones: list[int],
    bandpass_volume: float,
    background_volume: float,
    rng_seed: int,
    fade_ms: float,
    windows_dir: Path,
    spectra_dir: Path,
    file_tag: str
) -> np.ndarray:
    """Concatenate 8 window-based notes into one stereo stimulus and save per-note plots."""
    assert len(melody_L_semitones) == 8 and len(melody_R_semitones) == 8

    note_n = int(round(fs * note_dur_s))
    delay_samples = int(round((delay_ms / 1000.0) * fs))
    rng = np.random.default_rng(int(rng_seed))

    notes = []
    for i in range(8):
        bc1 = semitones_to_hz(base_center_hz, melody_L_semitones[i])
        bc2 = semitones_to_hz(base_center_hz, melody_R_semitones[i])

        bc1 = float(np.clip(bc1, 20.0, fs / 2.0))
        bc2 = float(np.clip(bc2, 20.0, fs / 2.0))

        # Prefix keeps your original naming scheme but adds file+note specificity
        plot_prefix = f"{file_tag}_note{i+1:02d}_L{int(round(bc1))}Hz_R{int(round(bc2))}Hz"

        note_stereo = make_one_note_window_based(
            fs=fs,
            note_n=note_n,
            delay_samples=delay_samples,
            bandCentre1_hz=bc1,
            bandCentre2_hz=bc2,
            bandpass_volume=bandpass_volume,
            background_volume=background_volume,
            rng=rng,
            fade_ms=fade_ms,
            windows_dir=windows_dir,
            spectra_dir=spectra_dir,
            plot_prefix=plot_prefix
        )
        notes.append(note_stereo)

    stereo = np.vstack(notes)

    # Normalize (like your to_int16 does later)
    peak = np.max(np.abs(stereo)) if stereo.size else 1.0
    if peak > 0 and np.isfinite(peak):
        stereo = stereo / peak

    return stereo


def main():
    # --- Parameters ---
    fs = 44100
    note_dur_s = 0.35
    delay_ms = 0.6
    bandpass_volume = 1.0
    background_volume = 1.0
    fade_ms = 10.0

    # Five base centers => five output wavs
    base_centers_hz = [250, 500, 1000, 2000, 4000]

    # Pentatonic mode
    degrees = PENTA_MAJOR  # switch to PENTA_MINOR if you want minor pentatonic
    mode_name = "majorPent" if degrees == PENTA_MAJOR else "minorPent"

    # 5 different pentatonic melody pairs (8 notes each)
    L_idx_sequences = [
        [0, 1, 2, 3, 4, 3, 2, 1],
        [0, 2, 4, 2, 3, 1, 2, 0],
        [0, 1, 3, 2, 4, 2, 1, 0],
        [0, 3, 2, 4, 3, 1, 2, 0],
        [0, 2, 1, 3, 2, 4, 3, 1],
    ]
    R_idx_sequences = [
        [3, 2, 1, 0, 1, 2, 3, 4],
        [2, 1, 0, 1, 2, 3, 2, 1],
        [4, 3, 1, 2, 0, 2, 3, 4],
        [1, 0, 2, 1, 3, 2, 4, 3],
        [2, 4, 3, 1, 2, 0, 1, 3],
    ]

    # --- Output folders (your original style) ---
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_outdir = Path(__file__).resolve().parent / "[to_analyze]"
    outdir = base_outdir / timestamp
    outdir.mkdir(parents=True, exist_ok=True)

    wavs_dir = outdir / "wavs"
    plots_dir = outdir / "plots"
    windows_dir = plots_dir / "windows"
    spectra_dir = plots_dir / "spectra"
    for d in (wavs_dir, plots_dir, windows_dir, spectra_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"\nRendered WAVs will be saved to: {wavs_dir}")
    print(f"Plots will be saved to: {plots_dir}\n")

    rendered_log = []

    # Render 5 WAVs
    for idx, base_hz in enumerate(base_centers_hz, start=1):
        melody_L = build_pentatonic_melody(degrees, L_idx_sequences[idx - 1])
        melody_R = build_pentatonic_melody(degrees, R_idx_sequences[idx - 1])

        file_tag = f"file{idx:02d}_{mode_name}_base{int(base_hz)}Hz"

        stereo = render_one_file_two_melodies(
            fs=fs,
            note_dur_s=note_dur_s,
            delay_ms=delay_ms,
            base_center_hz=float(base_hz),
            melody_L_semitones=melody_L,
            melody_R_semitones=melody_R,
            bandpass_volume=bandpass_volume,
            background_volume=background_volume,
            rng_seed=1000 + idx,
            fade_ms=fade_ms,
            windows_dir=windows_dir,
            spectra_dir=spectra_dir,
            file_tag=file_tag
        )

        fname = f"melody_{idx:02d}_{mode_name}_base{int(base_hz)}Hz_L8_R8.wav"
        wav_path = wavs_dir / fname
        wavwrite(wav_path, fs, to_int16(stereo))
        print(f"Saved: {wav_path}")

        rendered_log.append((fname, base_hz, melody_L, melody_R))

    # Parameters snapshot (same spirit + method description)
    delay_samples = int(round((delay_ms / 1000.0) * fs))
    params_lines = [
        f"Run timestamp: {timestamp}",
        f"Output folder: {outdir}",
        "",
        "[Signal & Timing]",
        f"fs (Hz): {fs}",
        f"note_dur_s (s): {note_dur_s}",
        f"notes_per_melody: 8",
        f"total_duration_per_file (s): {note_dur_s * 8:.3f}",
        f"delay_ms: {delay_ms}",
        f"delay_samples (@fs): {delay_samples}",
        f"fade_ms: {fade_ms}",
        "",
        "[Window Method (matches your original)]",
        "bandwidth_hz = bandCentre / 20",
        "bin_width_hz = fs / note_n",
        "bandwidth_bins = bandwidth_hz / bin_width_hz",
        "std_bins = bandwidth_bins / 2.355",
        "window = exp(-0.5 * ((bin-center)/std_bins)^2)",
        "complementaryWindow = 1 - window",
        "RIGHT: delayed bandpass (np.roll +delay_samples), advanced notch (np.roll -delay_samples)",
        "MIX: left = bp1+nt1, right = delayed(bp2)+advanced(nt2)",
        "",
        "[Levels]",
        f"bandpass_volume: {bandpass_volume}",
        f"background_volume: {background_volume}",
        "",
        "[Pentatonic]",
        f"mode: {mode_name}",
        f"degrees (semitones): {degrees}",
        "",
        "[Rendered files + melodies (semitone offsets from base)]",
    ]
    for fname, base_hz, mL, mR in rendered_log:
        params_lines += [
            f"{fname}",
            f"  base_center_hz: {base_hz}",
            f"  L semitones: {mL}",
            f"  R semitones: {mR}",
            ""
        ]

    (outdir / "00_run_parameters.txt").write_text("\n".join(params_lines), encoding="utf-8")
    print(f"\nWrote parameters snapshot to: {outdir / '00_run_parameters.txt'}")

    # Optional playback: play ~3 seconds of first stimulus
    if HAVE_SD:
        print("\nOptional playback: playing ~3 seconds of the FIRST rendered stimulus...")
        first_wav = wavs_dir / f"melody_01_{mode_name}_base{int(base_centers_hz[0])}Hz_L8_R8.wav"
        # Re-render first quickly for playback
        melody_L = build_pentatonic_melody(degrees, L_idx_sequences[0])
        melody_R = build_pentatonic_melody(degrees, R_idx_sequences[0])
        # (No plots here — just quick playback)
        rng = np.random.default_rng(1001)
        note_n = int(round(fs * note_dur_s))
        delay_samples = int(round((delay_ms / 1000.0) * fs))
        parts = []
        for i in range(8):
            bc1 = np.clip(semitones_to_hz(base_centers_hz[0], melody_L[i]), 20.0, fs/2)
            bc2 = np.clip(semitones_to_hz(base_centers_hz[0], melody_R[i]), 20.0, fs/2)
            # minimal generation (same pipeline)
            noise = rng.normal(0, 1, note_n)
            fft_noise = np.fft.rfft(noise)
            M = len(fft_noise)
            bin_width_hz = fs / note_n
            bw1 = bc1 / 20.0; bw2 = bc2 / 20.0
            std1 = (bw1 / bin_width_hz) / 2.355
            std2 = (bw2 / bin_width_hz) / 2.355
            cbin1 = freq_to_bin(bc1, bin_width_hz, fs)
            cbin2 = freq_to_bin(bc2, bin_width_hz, fs)
            w1 = gaussian_at_bin(cbin1, std1, M)
            w2 = gaussian_at_bin(cbin2, std2, M)
            cw1 = 1.0 - w1
            cw2 = 1.0 - w2
            bp1 = np.fft.irfft(fft_noise * w1) * bandpass_volume
            bp2 = np.fft.irfft(fft_noise * w2) * bandpass_volume
            nt1 = np.fft.irfft(fft_noise * cw1) * background_volume
            nt2 = np.fft.irfft(fft_noise * cw2) * background_volume
            bp2d = np.roll(bp2, delay_samples)
            nt2a = np.roll(nt2, -delay_samples)
            L = apply_fade(bp1 + nt1, fs, fade_ms=fade_ms)
            R = apply_fade(bp2d + nt2a, fs, fade_ms=fade_ms)
            parts.append(np.column_stack([L, R]))
        stereo = np.vstack(parts)
        stereo = _safe_norm(stereo)
        play_n = min(len(stereo), int(3.0 * fs))
        sd.play(stereo[:play_n], fs)
        sd.wait()
        print(f"Playback done. (File on disk: {first_wav})")

    print("\nDone.")


if __name__ == "__main__":
    main()