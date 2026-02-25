#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Window-Based Dichotic Pitch (Melody Pack @ Base 250 Hz) — FAST PLOTS (ONCE PER FILE)
-----------------------------------------------------------------------------------
This version keeps your exact window-based dichotic pitch method BUT:
✅ Generates the *same* plot set (windows full/zoom/log + spectra full/zoom/log)
✅ Only ONCE PER FILE (instead of per note), using a representative note:
    - PLOT_REPRESENTATIVE = "note1"   (first note)
    - PLOT_REPRESENTATIVE = "median"  (median note, for 8 notes -> note 4)

Everything else remains:
- 5 stereo WAV files, all base_center_hz = 250 Hz
- 8 notes per file
- Each note uses your window-based method:
  noise -> RFFT -> Gaussian window + complementary notch -> IRFFT
  right bandpass delayed, right notch advanced
  mix left = bp1+nt1, right = delayed(bp2)+advanced(nt2)
- Melody A is randomly assigned to Left or Right per run per file
- Console progress bar with mini progress:
  "File 2/5 | Note 6/8 | Plot 11/14 | ..."

Melody set (5 files):
1) Pentatonic scale melody (8 notes)
2) Major scale melody (8 notes)
3) Random melody (8 notes, constrained to major scale degrees)
4) Mary Had a Little Lamb (first phrase, padded to 8 notes) — public domain
5) Pop-chorus-like melody (NOT the exact "Never Gonna Give You Up" melody; copyrighted)

Dependencies:
    pip install numpy scipy matplotlib
Optional:
    pip install sounddevice
"""

import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write as wavwrite

# Optional playback
try:
    import sounddevice as sd
    HAVE_SD = True
except Exception:
    HAVE_SD = False


# =========================
# CONFIG: plot once per file
# =========================
PLOT_REPRESENTATIVE = "median"  # "note1" or "median"


# =========================
# Progress Bar + Run State
# =========================
class Progress:
    def __init__(self, total: int, title: str = "Working", width: int = 40):
        self.total = max(int(total), 1)
        self.title = title
        self.width = max(int(width), 10)
        self.done = 0
        self.start_time = time.time()
        sys.stdout.write(f"{self.title}\n")
        sys.stdout.flush()
        self._render("Starting...")

    def _render(self, msg: str):
        frac = self.done / self.total
        filled = int(round(self.width * frac))
        bar = "█" * filled + "░" * (self.width - filled)
        elapsed = time.time() - self.start_time
        eta = (elapsed * (self.total - self.done) / self.done) if self.done > 0 else 0.0
        line = (
            f"\r[{bar}] {self.done}/{self.total}  {frac*100:6.2f}%"
            f"  elapsed {elapsed:6.1f}s  ETA {eta:6.1f}s  | {msg}"
        )
        sys.stdout.write(line + " " * 10)
        sys.stdout.flush()

    def step(self, msg: str = ""):
        self.done = min(self.done + 1, self.total)
        self._render(msg)

    def info(self, msg: str):
        self._render(msg)

    def finish(self, msg: str = "Done"):
        self.done = self.total
        self._render(msg)
        sys.stdout.write("\n")
        sys.stdout.flush()


PROG = None

class RunState:
    """
    Tracks:
      File X/5
      Note Y/8
      Plot Z/14  (plots are per-file now; Z increments only when saving the 14 images)
    """
    def __init__(self, files_total: int, notes_per_file: int = 8, plots_per_file: int = 14):
        self.files_total = int(files_total)
        self.notes_per_file = int(notes_per_file)
        self.plots_per_file = int(plots_per_file)

        self.file_idx = 0
        self.note_idx = 0
        self.plot_idx = 0

    def set_file(self, file_idx_1based: int):
        self.file_idx = int(file_idx_1based)
        self.plot_idx = 0  # reset plots at file start

    def set_note(self, note_idx_1based: int):
        self.note_idx = int(note_idx_1based)

    def bump_plot(self):
        self.plot_idx = min(self.plot_idx + 1, self.plots_per_file)

    def prefix(self) -> str:
        return f"File {self.file_idx}/{self.files_total} | Note {self.note_idx}/{self.notes_per_file} | Plot {self.plot_idx}/{self.plots_per_file}"

STATE = None


# =========================
# Small helpers
# =========================
def to_int16(x: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(x)) if np.size(x) else 0.0
    if peak == 0 or not np.isfinite(peak):
        peak = 1.0
    return (x / peak * 32767.0).astype(np.int16)

def safe_norm(x: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(x)) if np.size(x) else 0.0
    return x / (peak if peak > 0 else 1.0)

def semitones_to_hz(base_hz: float, semitones: float) -> float:
    return float(base_hz) * (2.0 ** (float(semitones) / 12.0))

def apply_fade(note: np.ndarray, fs: int, fade_ms: float = 10.0) -> np.ndarray:
    n = len(note)
    fade_n = int(round((fade_ms / 1000.0) * fs))
    fade_n = max(1, min(fade_n, n // 2))
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, fade_n))
    out = note.copy()
    out[:fade_n] *= ramp
    out[-fade_n:] *= ramp[::-1]
    return out


# =========================
# Window-based method helpers
# =========================
def freq_to_bin(f_hz: float, bin_width_hz: float, fs: int) -> int:
    f_hz = max(0.0, min(float(f_hz), fs / 2.0))
    return int(round(f_hz / bin_width_hz))

def gaussian_at_bin(center_bin: int, sigma_bins: float, num_bins: int) -> np.ndarray:
    n = np.arange(num_bins)
    sigma_bins = max(float(sigma_bins), 1e-9)
    return np.exp(-0.5 * ((n - center_bin) / sigma_bins) ** 2)


# =========================
# Plotting (full/zoom/log) — SAME SET, ONCE PER FILE
# =========================
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
    w_left_bp  = window1 / (np.max(window1) if np.max(window1) > 0 else 1.0)
    w_left_nt  = complementaryWindow1 / (np.max(complementaryWindow1) if np.max(complementaryWindow1) > 0 else 1.0)
    w_right_bp = window2 / (np.max(window2) if np.max(window2) > 0 else 1.0)
    w_right_nt = complementaryWindow2 / (np.max(complementaryWindow2) if np.max(complementaryWindow2) > 0 else 1.0)

    def _saved(kind: str):
        if STATE: STATE.bump_plot()
        if PROG and STATE: PROG.step(f"{STATE.prefix()} | {kind} saved")

    # 02a overview linear
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
    _saved("windows plot")

    # 02b overview log-x
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
    _saved("windows plot")

    # 02c left linear full
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
    _saved("windows plot")

    # 02d left zoom
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
    _saved("windows plot")

    # 02e left log-x
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
    _saved("windows plot")

    # 02f right linear full
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
    _saved("windows plot")

    # 02g right zoom
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
    _saved("windows plot")

    # 02h right log-x
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
    _saved("windows plot")


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
    fft_left = np.fft.rfft(mixedl)
    fft_right = np.fft.rfft(mixedr)
    magL = np.abs(fft_left)
    magR = np.abs(fft_right)
    left_mag_norm = magL / (np.max(magL) if np.max(magL) > 0 else 1.0)
    right_mag_norm = magR / (np.max(magR) if np.max(magR) > 0 else 1.0)

    w_left_bp = window1 / (np.max(window1) if np.max(window1) > 0 else 1.0)
    w_left_nt = complementaryWindow1 / (np.max(complementaryWindow1) if np.max(complementaryWindow1) > 0 else 1.0)
    w_right_bp = window2 / (np.max(window2) if np.max(window2) > 0 else 1.0)
    w_right_nt = complementaryWindow2 / (np.max(complementaryWindow2) if np.max(complementaryWindow2) > 0 else 1.0)

    def _saved(kind: str):
        if STATE: STATE.bump_plot()
        if PROG and STATE: PROG.step(f"{STATE.prefix()} | {kind} saved")

    # 02 filters overview
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
    _saved("spectra plot")

    # 03 left full
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
    _saved("spectra plot")

    # 03 left zoom
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
    _saved("spectra plot")

    # 03 left log-x
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
    _saved("spectra plot")

    # 04 right full
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
    _saved("spectra plot")

    # 04 right zoom
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
    _saved("spectra plot")

    # 04 right log-x
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
    _saved("spectra plot")


# =========================
# One NOTE using exact pipeline (plots optional)
# =========================
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
    plot_prefix: str,
    do_plots: bool
) -> np.ndarray:
    if PROG and STATE:
        PROG.info(f"{STATE.prefix()} | generating noise + FFT")

    noise = rng.normal(0, 1, note_n)
    fft_noise = np.fft.rfft(noise)
    M = len(fft_noise)
    freqs = np.fft.rfftfreq(note_n, 1/fs)
    bin_width_hz = fs / note_n

    # EXACT bandwidth rule: center/20
    bandwidth1_hz = bandCentre1_hz / 20.0
    bandwidth2_hz = bandCentre2_hz / 20.0
    bandwidth1_bins = bandwidth1_hz / bin_width_hz
    bandwidth2_bins = bandwidth2_hz / bin_width_hz

    # EXACT std conversion
    std1 = bandwidth1_bins / 2.355
    std2 = bandwidth2_bins / 2.355

    cbin1 = freq_to_bin(bandCentre1_hz, bin_width_hz, fs)
    cbin2 = freq_to_bin(bandCentre2_hz, bin_width_hz, fs)

    window1 = gaussian_at_bin(cbin1, std1, M)
    window2 = gaussian_at_bin(cbin2, std2, M)

    complementaryWindow1 = 1.0 - window1
    complementaryWindow2 = 1.0 - window2

    # Plots ONCE PER FILE (representative note)
    if do_plots:
        if PROG and STATE:
            PROG.info(f"{STATE.prefix()} | saving representative plots")
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

    if PROG and STATE:
        PROG.info(f"{STATE.prefix()} | filtering + mixing")

    # bandpass
    bandpassed1 = np.fft.irfft(fft_noise * window1) * bandpass_volume
    bandpassed2 = np.fft.irfft(fft_noise * window2) * bandpass_volume

    # delay right bandpass
    delayed_bandpassed2 = np.roll(bandpassed2, delay_samples)

    # notch (complementary)
    notched1 = np.fft.irfft(fft_noise * complementaryWindow1) * background_volume
    notched2 = np.fft.irfft(fft_noise * complementaryWindow2) * background_volume

    # advance right notch
    advanced_notched2 = np.roll(notched2, -delay_samples)

    mixedl = bandpassed1 + notched1
    mixedr = delayed_bandpassed2 + advanced_notched2

    mixedl = apply_fade(mixedl, fs, fade_ms=fade_ms)
    mixedr = apply_fade(mixedr, fs, fade_ms=fade_ms)

    if do_plots:
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


# =========================
# Render one file: 8 notes
# =========================
def representative_note_index(n_notes: int) -> int:
    """0-based index for representative note."""
    if PLOT_REPRESENTATIVE.lower() == "note1":
        return 0
    # median: for even N=8 -> choose lower median -> 4th note => index 3
    return (n_notes - 1) // 2  # 8 -> 3

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
    assert len(melody_L_semitones) == 8 and len(melody_R_semitones) == 8

    n_notes = 8
    rep_idx = representative_note_index(n_notes)

    note_n = int(round(fs * note_dur_s))
    delay_samples = int(round((delay_ms / 1000.0) * fs))
    rng = np.random.default_rng(int(rng_seed))

    notes = []
    for i in range(n_notes):
        if STATE:
            STATE.set_note(i + 1)
        if PROG and STATE:
            PROG.info(f"{STATE.prefix()} | generating note")

        bc1 = float(np.clip(semitones_to_hz(base_center_hz, melody_L_semitones[i]), 20.0, fs / 2.0))
        bc2 = float(np.clip(semitones_to_hz(base_center_hz, melody_R_semitones[i]), 20.0, fs / 2.0))

        do_plots = (i == rep_idx)
        # plots are per file, so prefix includes which note we used
        plot_prefix = f"{file_tag}_REPnote{i+1:02d}_L{int(round(bc1))}Hz_R{int(round(bc2))}Hz" if do_plots else "unused"

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
            plot_prefix=plot_prefix,
            do_plots=do_plots
        )

        notes.append(note_stereo)

        if PROG and STATE:
            PROG.step(f"{STATE.prefix()} | note complete")

    stereo = np.vstack(notes)
    peak = np.max(np.abs(stereo)) if stereo.size else 1.0
    if peak > 0 and np.isfinite(peak):
        stereo = stereo / peak
    return stereo


# =========================
# Melody definitions (8 notes each)
# =========================
MAJOR_SCALE_8 = [0, 2, 4, 5, 7, 9, 11, 12]
PENTA_MELODY_8 = [0, 2, 4, 7, 9, 7, 4, 2]

# Mary Had a Little Lamb (first phrase), in C major: E D C D E E E (pad with D)
MARY_FIRST_PHRASE_8 = [4, 2, 0, 2, 4, 4, 4, 2]

def random_major_melody_8(rng: np.random.Generator) -> list[int]:
    degrees = [0, 2, 4, 5, 7, 9, 11, 12]
    out = [0]
    for _ in range(7):
        prev = out[-1]
        close = sorted(degrees, key=lambda d: abs(d - prev))[:4]
        out.append(int(rng.choice(close)))
    return out

# NOT the copyrighted Rick Astley melody (generic pop-like contour)
POP_CHORUS_LIKE_8 = [7, 7, 9, 11, 9, 7, 5, 4]

def tonic_drone_8() -> list[int]:
    return [0, 0, 0, 0, 0, 0, 0, 0]


# =========================
# MAIN
# =========================
def main():
    global PROG, STATE

    fs = 44100
    note_dur_s = 0.35
    delay_ms = 0.6
    fade_ms = 10.0
    bandpass_volume = 1.0
    background_volume = 1.0

    # Base always 250 Hz
    base_center_hz = 250.0

    # 5 files = 5 melody types
    melody_specs = [
        ("01_pentatonic", PENTA_MELODY_8),
        ("02_major_scale", MAJOR_SCALE_8),
        ("03_random_major", None),
        ("04_mary_lamb", MARY_FIRST_PHRASE_8),
        ("05_pop_chorus_like", POP_CHORUS_LIKE_8),
    ]

    melody_B = tonic_drone_8()

    # Output folders
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_outdir = Path(__file__).resolve().parent / "[to_analyze]"
    outdir = base_outdir / timestamp
    wavs_dir = outdir / "wavs"
    plots_dir = outdir / "plots"
    windows_dir = plots_dir / "windows"
    spectra_dir = plots_dir / "spectra"
    for d in (wavs_dir, windows_dir, spectra_dir):
        d.mkdir(parents=True, exist_ok=True)

    FILES_TOTAL = len(melody_specs)
    NOTES_PER_FILE = 8
    PLOTS_PER_FILE = 14  # plots happen once per file now

    STATE = RunState(files_total=FILES_TOTAL, notes_per_file=NOTES_PER_FILE, plots_per_file=PLOTS_PER_FILE)

    # Total steps: plots (files*14) + note completes (files*8) + wav saves (files) + some slack
    total_steps = (FILES_TOTAL * PLOTS_PER_FILE) + (FILES_TOTAL * NOTES_PER_FILE) + FILES_TOTAL + 12
    PROG = Progress(total_steps, title="Generating window-based dichotic pitch melody pack (fast plots)...", width=40)
    PROG.step("Output folders created")

    print(f"\nOutput:\n  WAVs:  {wavs_dir}\n  Plots: {plots_dir}\n")
    print(f"Plot mode: ONCE PER FILE using representative '{PLOT_REPRESENTATIVE}' note.\n")

    # Randomization seed
    run_seed = int(datetime.now().timestamp() * 1000) % (2**32)
    run_rng = np.random.default_rng(run_seed)
    PROG.step("Randomization seeded")

    rendered_log = []

    for idx, (label, melody_A_template) in enumerate(melody_specs, start=1):
        STATE.set_file(idx)
        if PROG and STATE:
            PROG.info(f"{STATE.prefix()} | starting file {label}")

        if melody_A_template is None:
            melody_A = random_major_melody_8(run_rng)
        else:
            melody_A = list(melody_A_template)

        # Random ear assignment for Melody A (per file per run)
        A_on_left = bool(run_rng.integers(0, 2))
        if A_on_left:
            melody_L = melody_A
            melody_R = melody_B
            assignment_tag = "AonL"
        else:
            melody_L = melody_B
            melody_R = melody_A
            assignment_tag = "AonR"

        file_tag = f"file{idx:02d}_{label}_base250Hz_{assignment_tag}"

        stereo = render_one_file_two_melodies(
            fs=fs,
            note_dur_s=note_dur_s,
            delay_ms=delay_ms,
            base_center_hz=base_center_hz,
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

        fname = f"melody_{idx:02d}_{label}_base250Hz_L8_R8_{assignment_tag}.wav"
        wav_path = wavs_dir / fname
        wavwrite(wav_path, fs, to_int16(stereo))
        if PROG and STATE:
            PROG.step(f"{STATE.prefix()} | WAV saved: {fname}")

        rendered_log.append({
            "fname": fname,
            "label": label,
            "base_hz": base_center_hz,
            "run_seed": run_seed,
            "assignment_tag": assignment_tag,
            "melody_A_semitones": melody_A,
            "melody_B_semitones": melody_B,
            "melody_L_actual": melody_L,
            "melody_R_actual": melody_R
        })

    # Write parameter snapshot
    delay_samples = int(round((delay_ms / 1000.0) * fs))
    params_lines = [
        f"Run timestamp: {timestamp}",
        f"Output folder: {outdir}",
        "",
        "[Signal & Timing]",
        f"fs (Hz): {fs}",
        f"note_dur_s (s): {note_dur_s}",
        f"notes_per_melody: {NOTES_PER_FILE}",
        f"total_duration_per_file (s): {note_dur_s * NOTES_PER_FILE:.3f}",
        f"delay_ms: {delay_ms}",
        f"delay_samples (@fs): {delay_samples}",
        f"fade_ms: {fade_ms}",
        "",
        "[Base]",
        f"base_center_hz: {base_center_hz}",
        "",
        "[Plotting]",
        "Plots generated ONCE PER FILE (representative note).",
        f"PLOT_REPRESENTATIVE: {PLOT_REPRESENTATIVE}",
        "If 'median' and 8 notes: representative note = note 4.",
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
        "[Randomization]",
        f"run_seed (ear assignment + random melody RNG): {run_seed}",
        "assignment_tag: AonL means Melody A on Left; AonR means Melody A on Right",
        "",
        "[Notes]",
        "Mary Had a Little Lamb: first phrase (public domain), padded to 8 notes",
        "Pop chorus-like: NOT the exact 'Never Gonna Give You Up' melody (copyrighted)",
        "",
        "[Rendered files + semitone offsets from base]",
    ]
    for item in rendered_log:
        params_lines += [
            f"{item['fname']}",
            f"  label: {item['label']}",
            f"  base_center_hz: {item['base_hz']}",
            f"  assignment: {item['assignment_tag']}",
            f"  Melody A semitones: {item['melody_A_semitones']}",
            f"  Melody B semitones: {item['melody_B_semitones']}",
            f"  ACTUAL L semitones: {item['melody_L_actual']}",
            f"  ACTUAL R semitones: {item['melody_R_actual']}",
            ""
        ]

    (outdir / "00_run_parameters.txt").write_text("\n".join(params_lines), encoding="utf-8")
    PROG.step("Wrote 00_run_parameters.txt")

    # Optional playback (first file, ~3 seconds)
    if HAVE_SD:
        PROG.info("Optional playback: ~3 seconds of first stimulus...")
        first = rendered_log[0]
        # quick regenerate without plots
        rng = np.random.default_rng(1001)
        note_n = int(round(fs * note_dur_s))
        delay_samples = int(round((delay_ms / 1000.0) * fs))
        parts = []
        for i in range(8):
            bc1 = float(np.clip(semitones_to_hz(base_center_hz, first["melody_L_actual"][i]), 20.0, fs/2))
            bc2 = float(np.clip(semitones_to_hz(base_center_hz, first["melody_R_actual"][i]), 20.0, fs/2))
            noise = rng.normal(0, 1, note_n)
            fft_noise = np.fft.rfft(noise)
            M = len(fft_noise)
            bin_width_hz = fs / note_n
            bw1 = bc1 / 20.0
            bw2 = bc2 / 20.0
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
        stereo_play = safe_norm(np.vstack(parts))
        play_n = min(len(stereo_play), int(3.0 * fs))
        sd.play(stereo_play[:play_n], fs)
        sd.wait()
        PROG.step("Playback done")
    else:
        PROG.step("Playback skipped (sounddevice not installed)")

    PROG.finish("All files + plots generated.")


if __name__ == "__main__":
    main()