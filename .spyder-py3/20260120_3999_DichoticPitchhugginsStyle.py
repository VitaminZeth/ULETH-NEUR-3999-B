import numpy as np
from scipy.io.wavfile import write as wavwrite

# =========================
# Parameters
# =========================
fs = 48000          # Sampling rate (Hz)
duration = 1.0      # Duration (seconds)
f_low = 400         # Lower edge of phase-shift band (Hz)
f_high = 600        # Upper edge of phase-shift band (Hz)
phase_shift = np.pi # 180° phase shift in the band

# =========================
# Helper: Normalize to int16
# =========================
def to_int16(x):
    # Avoid division by zero just in case
    max_val = np.max(np.abs(x))
    if max_val < 1e-9:
        return (x * 0).astype(np.int16)
    x_norm = x / max_val
    return (x_norm * 32767).astype(np.int16)

# =========================
# 1) Generate base broadband noise
# =========================
n_samples = int(fs * duration)
mono_noise = np.random.randn(n_samples)

# Diotic reference: same noise to both ears
diotic_stereo = np.column_stack([mono_noise, mono_noise])

# =========================
# 2) Create Huggins-style dichotic pitch
#    - Left ear: original noise
#    - Right ear: same noise, but with a phase shift in a narrow band
# =========================

# FFT of the mono noise (for right ear manipulation)
X = np.fft.rfft(mono_noise)
freqs = np.fft.rfftfreq(n_samples, 1 / fs)

# Find frequency bin indices inside the band
band_mask = (freqs >= f_low) & (freqs <= f_high)

# Apply a constant phase shift ONLY in that band for the right ear
X_mod = X.copy()
X_mod[band_mask] *= np.exp(1j * phase_shift)

# Inverse FFT to get time-domain right-ear signal
right_ear = np.fft.irfft(X_mod, n_samples)

# Left ear is the original noise
left_ear = mono_noise.copy()

# Stack into stereo: [left, right]
dichotic_stereo = np.column_stack([left_ear, right_ear])

# =========================
# 3) Save WAV files
# =========================
wavwrite("diotic_noise.wav", fs, to_int16(diotic_stereo))
wavwrite("dichotic_pitch.wav", fs, to_int16(dichotic_stereo))

print("Saved 'diotic_noise.wav' and 'dichotic_pitch.wav'.")
print(f"Dichotic pitch band: {f_low}-{f_high} Hz (center ≈ {(f_low + f_high) / 2:.1f} Hz)")
