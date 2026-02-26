# ==== AUDIO BACKEND SETUP ====
from psychopy import prefs
prefs.hardware['audioLib'] = ['PTB']  # Use high-precision backend for EEG

# ==== IMPORTS ====
from psychopy import sound, core
import numpy as np
from scipy.fftpack import fft, ifft
from scipy.io.wavfile import write

# ==== PARAMETERS ====
sample_rate = 48000
duration = 1.0  # seconds
n_samples = int(sample_rate * duration)
center_freq = 400
bandwidth = 20

# ==== GENERATE WHITE NOISE ====
np.random.seed(0)
white_noise = np.random.normal(0, 1, n_samples)

# ==== FFT: TO FREQUENCY DOMAIN ====
freq_noise = fft(white_noise)
freqs = np.fft.fftfreq(n_samples, d=1/sample_rate)

# ==== PHASE SHIFT: ONLY RIGHT CHANNEL ====
freq_noise_shifted = freq_noise.copy()
for i, f in enumerate(freqs):
    if center_freq - bandwidth < abs(f) < center_freq + bandwidth:
        freq_noise_shifted[i] *= np.exp(1j * np.pi)

# ==== IFFT: BACK TO TIME DOMAIN ====
right_noise = np.real(ifft(freq_noise_shifted))

# ==== NORMALIZE AND COMBINE STEREO ====
max_val = max(np.max(np.abs(white_noise)), np.max(np.abs(right_noise)))
left = white_noise / max_val
right = right_noise / max_val
stereo = np.vstack((left, right)).T.astype(np.float32)  # PTB prefers float32

# ==== PLAY WITH PTB ====
stim = sound.Sound(value=stereo, sampleRate=sample_rate, stereo=True)
stim.play()
core.wait(duration + 0.2)  # wait slightly longer to ensure complete playback

# ==== SAVE (Optional) ====
write("dichotic_pitch.wav", sample_rate, (stereo * 32767).astype(np.int16))
