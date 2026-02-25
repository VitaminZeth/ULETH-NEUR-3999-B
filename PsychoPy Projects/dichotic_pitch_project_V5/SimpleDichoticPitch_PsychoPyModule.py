#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PsychoPy-friendly dichotic pitch / Huggins-pitch style stimulus generator.

Based on your SimpleDichoticPitchDemo.py, but refactored to:
- avoid sounddevice playback (PsychoPy will handle audio),
- avoid plotting / filesystem side-effects unless you ask for saving,
- provide a single function `generate_dichotic_pitch_stim()` that returns a
  stereo float32 array in [-1, 1].

Usage patterns:
1) Generate in a Code Component each trial and feed to a Sound object.
2) Or pre-render WAVs (recommended for tight timing / EEG) and play by filename.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from scipy.io.wavfile import write as wavwrite


# -------------------------
# Utilities
# -------------------------
def _safe_peak_norm(x: np.ndarray, peak_target: float = 0.999) -> np.ndarray:
    """Normalize to peak_target (<=1.0) to avoid clipping."""
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if not np.isfinite(peak) or peak <= 0:
        return x.astype(np.float32, copy=False)
    return (x / peak * peak_target).astype(np.float32, copy=False)


def _to_int16(x: np.ndarray) -> np.ndarray:
    """Convert float [-1,1] to int16 PCM with safe scaling."""
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16)


def _freq_to_bin(f_hz: float, bin_width_hz: float, fs: float) -> int:
    """Clamp frequency and map to nearest RFFT bin index."""
    f_hz = max(0.0, min(float(f_hz), float(fs) / 2.0))
    return int(round(f_hz / bin_width_hz))


def _gaussian_window(num_bins: int, center_bin: int, sigma_bins: float) -> np.ndarray:
    """Unit-peak Gaussian centered at center_bin over bins [0..num_bins-1]."""
    n = np.arange(num_bins, dtype=np.float64)
    sigma_bins = max(float(sigma_bins), 1e-12)
    return np.exp(-0.5 * ((n - float(center_bin)) / sigma_bins) ** 2)


def _integer_delay_no_wrap(x: np.ndarray, delay_samples: int) -> np.ndarray:
    """
    Delay signal by delay_samples WITHOUT wrap-around.
    Positive delay shifts right (later); negative shifts left (earlier).
    """
    delay_samples = int(delay_samples)
    if delay_samples == 0:
        return x.copy()
    y = np.zeros_like(x)
    if delay_samples > 0:
        y[delay_samples:] = x[:-delay_samples]
    else:
        d = -delay_samples
        y[:-d] = x[d:]
    return y


@dataclass
class DichoticPitchParams:
    fs: int = 44100
    duration_s: float = 1.0
    # Center frequencies (Hz) for the Gaussian band windows
    band_centre_left_hz: float = 500.0
    band_centre_right_hz: float = 500.0
    # Bandwidth as a fraction of centre (your demo used centre/20)
    bandwidth_fraction: float = 1.0 / 20.0
    # Interaural delay in ms applied to the RIGHT band-passed component
    delay_ms: float = 0.6
    # Levels
    bandpass_volume: float = 1.0
    background_volume: float = 1.0
    # Randomness
    seed: int | None = None  # None -> unpredictable each call


def generate_dichotic_pitch_stim(p: DichoticPitchParams) -> tuple[np.ndarray, dict]:
    """
    Generate a stereo dichotic stimulus.

    Returns
    -------
    stereo : np.ndarray
        Shape (n_samples, 2), float32 in [-1, 1] (peak normalized).
    info : dict
        Helpful metadata (bins, bandwidth, delay_samples, etc.).
    """
    fs = int(p.fs)
    n_samples = int(round(p.duration_s * fs))
    rng = np.random.default_rng(p.seed)

    # Noise
    noise = rng.normal(0.0, 1.0, n_samples).astype(np.float64)

    # FFT
    fft_noise = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(n_samples, 1.0 / fs)

    bin_width_hz = fs / n_samples
    bw_hz_left = float(p.band_centre_left_hz) * float(p.bandwidth_fraction)
    bw_hz_right = float(p.band_centre_right_hz) * float(p.bandwidth_fraction)

    # Convert FWHM (≈ bandwidth) to sigma (bins): sigma = FWHM/2.355
    bw_bins_left = bw_hz_left / bin_width_hz
    bw_bins_right = bw_hz_right / bin_width_hz
    sigma_left = max(bw_bins_left / 2.355, 1e-12)
    sigma_right = max(bw_bins_right / 2.355, 1e-12)

    M = len(fft_noise)
    cbin_left = _freq_to_bin(p.band_centre_left_hz, bin_width_hz, fs)
    cbin_right = _freq_to_bin(p.band_centre_right_hz, bin_width_hz, fs)

    window_left = _gaussian_window(M, cbin_left, sigma_left)
    window_right = _gaussian_window(M, cbin_right, sigma_right)

    notch_left = 1.0 - window_left
    notch_right = 1.0 - window_right

    # Band-pass components
    band_left = np.fft.irfft(fft_noise * window_left, n=n_samples) * float(p.bandpass_volume)
    band_right = np.fft.irfft(fft_noise * window_right, n=n_samples) * float(p.bandpass_volume)

    # Complementary notches (backgrounds)
    back_left = np.fft.irfft(fft_noise * notch_left, n=n_samples) * float(p.background_volume)
    back_right = np.fft.irfft(fft_noise * notch_right, n=n_samples) * float(p.background_volume)

    # Interaural delay: apply to RIGHT bandpass, and (optionally) "compensate" background.
    delay_samples = int(round((float(p.delay_ms) / 1000.0) * fs))
    band_right_delayed = _integer_delay_no_wrap(band_right, delay_samples)

    # Your original code advanced the right notch by -delay (no wrap). Keep that behavior here:
    back_right_advanced = _integer_delay_no_wrap(back_right, -delay_samples)

    left_mix = band_left + back_left
    right_mix = band_right_delayed + back_right_advanced

    stereo = np.column_stack([left_mix, right_mix]).astype(np.float32)
    stereo = _safe_peak_norm(stereo)  # peak normalize both channels together

    info = dict(
        fs=fs,
        n_samples=n_samples,
        duration_s=float(p.duration_s),
        delay_ms=float(p.delay_ms),
        delay_samples=delay_samples,
        bin_width_hz=float(bin_width_hz),
        band_centre_left_hz=float(p.band_centre_left_hz),
        band_centre_right_hz=float(p.band_centre_right_hz),
        cbin_left=int(cbin_left),
        cbin_right=int(cbin_right),
        bw_hz_left=float(bw_hz_left),
        bw_hz_right=float(bw_hz_right),
        sigma_left_bins=float(sigma_left),
        sigma_right_bins=float(sigma_right),
    )
    return stereo, info


def save_stim_wav(stereo: np.ndarray, fs: int, path: str | Path) -> Path:
    """Save a stereo float array [-1,1] as 16-bit PCM WAV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wavwrite(path, int(fs), _to_int16(stereo))
    return path


if __name__ == "__main__":
    # Quick smoke test: render a file next to this script
    p = DichoticPitchParams()
    stereo, info = generate_dichotic_pitch_stim(p)
    out = Path(__file__).with_name("dichotic_pitch_test.wav")
    save_stim_wav(stereo, p.fs, out)
    print("Saved:", out)
    print("Info:", info)
