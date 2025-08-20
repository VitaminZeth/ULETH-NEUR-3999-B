#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single‑file dichotic pitch generator (Dougherty et al., 1998)

Open this file in Anaconda/Spyder and Run. It will:
  • Generate example dichotic‑pitch stimuli using complementary Gaussian masks
    with SBR<=1 (no monaural cues), ITD on the signal band only, 1200 Hz LPF,
    and 50 ms half‑Gaussian ramps.
  • Save WAVs to a folder named "dichoticpitchstimuli" and print the path.
  • (Optional) Play audio if the 'sounddevice' library is available.

Author: ChatGPT — GPT‑5 Thinking | License: MIT
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence, Optional, Literal, Tuple

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.io.wavfile import write as wavwrite

# Optional playback (safe to run without it)
try:
    import sounddevice as sd  # type: ignore
    _HAVE_SD = True
except Exception:
    _HAVE_SD = False

# =============================
# Utility helpers
# =============================

def _ensure_outdir(name: str = "dichoticpitchstimuli") -> str:
    out = os.path.abspath(name)
    os.makedirs(out, exist_ok=True)
    return out


def _half_gaussian_ramp(n_samples: int, fs: float, ramp_ms: float = 50.0) -> np.ndarray:
    """Create fade‑in/out half‑Gaussian envelope of total length n_samples."""
    ramp = int(round((ramp_ms / 1000.0) * fs))
    if ramp <= 0:
        return np.ones(n_samples, dtype=float)
    # Use a half‑Gaussian shape normalized to [0,1]
    t = np.linspace(-3.0, 0.0, ramp)
    fade_in = np.exp(-(t**2) / 2.0)
    fade_in = (fade_in - fade_in.min()) / max(1e-12, (fade_in.max() - fade_in.min()))
    fade_out = fade_in[::-1]
    env = np.ones(n_samples, dtype=float)
    env[:ramp] *= fade_in
    env[-ramp:] *= fade_out
    return env


def _lowpass_sos(fs: float, cutoff_hz: float = 1200.0, order: int = 8):
    return butter(order, cutoff_hz, btype="low", fs=fs, output="sos")


# =============================
# Frequency‑domain Gaussian masks
# =============================

def gaussian_mask(freqs: np.ndarray, center_hz: float, spread_frac: float, sbr: float) -> np.ndarray:
    """Signal mask S(f) = sbr * exp(-((f-p)^2)/s^2) with s = spread_frac*center."""
    s = max(1e-9, spread_frac * center_hz)
    return sbr * np.exp(-((freqs - center_hz) ** 2) / (s**2))


def complementary_background(S: np.ndarray) -> np.ndarray:
    """Background mask B(f) = 1 - S(f) (for S<1), else 0; flattens monaural spectrum."""
    return np.where(S < 1.0, 1.0 - S, 0.0)


# =============================
# Core generator
# =============================

@dataclass
class SegmentSpec:
    duration_s: float
    center_hz: float
    spread_frac: float = 0.05  # ~5% of center
    sbr: float = 1.0           # <=1.0 avoids monaural cues
    itd_ms: float = 0.6        # signal ITD; background stays at 0 ms
    lead: Literal["left", "right"] = "right"


def make_dichotic_segment(spec: SegmentSpec, fs: float, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """One stereo segment implementing Dougherty et al. (1998).
    Returns float64 array (n, 2) roughly in [-1, 1].
    """
    if rng is None:
        rng = np.random.default_rng()

    n = int(round(spec.duration_s * fs))
    # Independent white‑noise sources for signal and background
    sig = rng.standard_normal(n)
    bg = rng.standard_normal(n)

    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    SIG = np.fft.rfft(sig)
    BG = np.fft.rfft(bg)

    S = gaussian_mask(freqs, spec.center_hz, spec.spread_frac, spec.sbr)
    B = complementary_background(S)

    SIG_f = SIG * S
    BG_f = BG * B

    sig_filt = np.fft.irfft(SIG_f, n)
    bg_filt = np.fft.irfft(BG_f, n)

    # ITD only on signal band; background centered
    itd_s = spec.itd_ms / 1000.0

    def _time_shift(x: np.ndarray, shift_s: float) -> np.ndarray:
        X = np.fft.rfft(x)
        phase = np.exp(-1j * 2 * np.pi * freqs * shift_s)
        return np.fft.irfft(X * phase, n)

    if spec.lead == "right":
        sig_L = _time_shift(sig_filt, +itd_s)
        sig_R = sig_filt
    else:
        sig_L = sig_filt
        sig_R = _time_shift(sig_filt, +itd_s)

    left = sig_L + bg_filt
    right = sig_R + bg_filt

    # 1200 Hz low‑pass and 50 ms ramps
    sos = _lowpass_sos(fs, 1200.0, order=8)
    left = sosfiltfilt(sos, left)
    right = sosfiltfilt(sos, right)

    env = _half_gaussian_ramp(n, fs, 50.0)
    left *= env
    right *= env

    stereo = np.vstack([left, right]).T
    peak = np.max(np.abs(stereo))
    if peak > 0:
        stereo = 0.95 * stereo / peak
    return stereo.astype(np.float64)


def make_sequence(
    centers_hz: Sequence[float],
    seg_duration_s: float,
    fs: float,
    spread_frac: float = 0.05,
    sbr: float = 1.0,
    itd_ms: float = 0.6,
    lead_policy: Literal["fixed_left", "fixed_right", "alternate", "random"] = "alternate",
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()

    segs = []
    for i, c in enumerate(centers_hz):
        if lead_policy == "fixed_right":
            lead = "right"
        elif lead_policy == "fixed_left":
            lead = "left"
        elif lead_policy == "alternate":
            lead = "right" if (i % 2 == 0) else "left"
        else:
            lead = "right" if rng.random() < 0.5 else "left"

        seg = make_dichotic_segment(
            SegmentSpec(seg_duration_s, c, spread_frac, sbr, itd_ms, lead),
            fs=fs,
            rng=rng,
        )
        segs.append(seg)
    return np.vstack(segs)


def write_wav_stereo(path: str, fs: int, stereo: np.ndarray) -> None:
    x = np.clip(stereo, -1.0, 1.0)
    wavwrite(path, int(fs), (x * 32767.0).astype(np.int16))


# =============================
# Predefined demos
# =============================

def pitch_identification_demo(fs: int = 44100, seg_ms: float = 200.0, sbr: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
    dur = seg_ms / 1000.0
    rising = make_sequence([400, 500, 600], dur, fs, spread_frac=0.05, sbr=sbr, itd_ms=0.6, lead_policy="alternate")
    falling = make_sequence([600, 500, 400], dur, fs, spread_frac=0.05, sbr=sbr, itd_ms=0.6, lead_policy="alternate")
    return rising, falling


def melody_localization_demo(fs: int = 44100, seg_ms: float = 200.0, sbr: float = 0.8) -> np.ndarray:
    dur = seg_ms / 1000.0
    centers = [440, 550, 495, 660]  # approximate melody centers
    return make_sequence(centers, dur, fs, spread_frac=0.05, sbr=sbr, itd_ms=0.6, lead_policy="random")


# =============================
# Main: run in Anaconda/Spyder
# =============================
if __name__ == "__main__":
    fs = 44100
    outdir = _ensure_outdir("dichoticpitchstimuli")
    print(f"Output directory: {outdir}")

    # Example 1: single 1 s segment at 500 Hz
    seg = make_dichotic_segment(SegmentSpec(1.0, 500.0, 0.05, 0.8, 0.6, "right"), fs)
    p1 = os.path.join(outdir, "dichotic_segment_500Hz.wav")
    write_wav_stereo(p1, fs, seg)
    print(f"Saved: {p1}")

    # Example 2: rising vs falling (3 x 200 ms)
    rise, fall = pitch_identification_demo(fs=fs, seg_ms=200, sbr=0.8)
    p2 = os.path.join(outdir, "dichotic_rising.wav")
    p3 = os.path.join(outdir, "dichotic_falling.wav")
    write_wav_stereo(p2, fs, rise)
    write_wav_stereo(p3, fs, fall)
    print(f"Saved: {p2}")
    print(f"Saved: {p3}")

    # Example 3: melody localization demo (4 x 200 ms)
    mel = melody_localization_demo(fs=fs, seg_ms=200, sbr=0.8)
    p4 = os.path.join(outdir, "dichotic_melody.wav")
    write_wav_stereo(p4, fs, mel)
    print(f"Saved: {p4}")

    # Optional playback (stereo)
    if _HAVE_SD:
        try:
            print("Playing: dichotic_segment_500Hz.wav …")
            sd.play(seg / max(1e-12, np.max(np.abs(seg))), fs)
            sd.wait()
        except Exception as e:
            print(f"Playback skipped: {e}")
    else:
        print("Tip: install 'sounddevice' to enable playback (conda install -c conda-forge python-sounddevice)")
