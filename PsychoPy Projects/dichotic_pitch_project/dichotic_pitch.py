# ==== AUDIO BACKEND SETUP ====
from psychopy import prefs
prefs.hardware['audioLib'] = ['sounddevice']  # Use a more compatible backend

# ==== IMPORTS ====
from psychopy import sound, core
import numpy as np
from scipy.fftpack import fft, ifft
from scipy.io.wavfile import write

# ==== PARAMETERS ====
sample_rate = 48000  # More broadly supported than 44100
duration = 1.0       # seconds
n_samples = int(sample_rate * duration)
center_freq = 400    # Frequency of pitch illusion
bandwidth = 20       # Width of frequency band to shift

# ==== GENERATE WHITE NOISE ====
np.random.seed(0)
white_noise = np.random.normal(0, 1, n_samples)

# ==== FFT: TO FREQUENCY DOMAIN ====
freq_noise = fft(white_noise)
freqs = np.fft.fftfreq(n_samples, d=1/sample_rate)

# ==== PHASE SHIFT RIGHT CHANNEL ONLY ====
freq_noise_shifted = freq_noise.copy()
for i, f in enumerate(freqs):
    if center_freq - bandwidth < abs(f) < center_freq + bandwidth:
        freq_noise_shifted[i] *= np.exp(1j * np.pi)  # 180° phase shift

# ==== IFFT: BACK TO TIME DOMAIN ====
right_noise = np.real(ifft(freq_noise_shifted))

# ==== NORMALIZE TO -1 to +1 RANGE ====
max_val = max(np.max(np.abs(white_noise)), np.max(np.abs(right_noise)))
left = white_noise / max_val
right = right_noise / max_val

# ==== COMBINE INTO STEREO ARRAY ====
