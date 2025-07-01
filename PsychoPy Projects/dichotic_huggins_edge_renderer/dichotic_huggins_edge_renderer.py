# PsychoPy: Render & Save Dichotic, Huggins, and Binaural Edge Pitch stimuli

from psychopy import prefs
prefs.hardware['audioLib'] = ['sounddevice']

from psychopy import sound, core, event, visual
import numpy as np
from scipy.fft import fft, ifft, fftfreq
from scipy.io.wavfile import write
import os
from datetime import datetime

# Settings
SAMPLE_RATE = 44100
DURATION = 1.0
N_SAMPLES = int(SAMPLE_RATE * DURATION)
FREQ = 600     # Center frequency for dichotic and Huggins
BANDWIDTH = 20 # Hz
EDGE_FREQ = 700 # For binaural edge

# Output folders
BASE_DIR = "output"
DP_DIR = os.path.join(BASE_DIR, "dichotic_pitch")
HP_DIR = os.path.join(BASE_DIR, "huggins_pitch")
BE_DIR = os.path.join(BASE_DIR, "binaural_edge_pitch")
for folder in [DP_DIR, HP_DIR, BE_DIR]:
    os.makedirs(folder, exist_ok=True)

# PsychoPy window
win = visual.Window(fullscr=False, color='black')
txt = visual.TextStim(win, color='white', height=0.08, wrapWidth=1.5)

def play_stim_and_save(stereo, label, outdir):
    snd = sound.Sound(value=stereo, sampleRate=SAMPLE_RATE, stereo=True)
    snd.setVolume(0.5)
    txt.text = f"{label}\n(Press SPACE to play)"
    txt.draw(); win.flip()
    event.waitKeys(keyList=['space'])
    snd.play(); core.wait(DURATION)
    # Save WAV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(outdir, f"{label.replace(' ', '_').lower()}_{timestamp}.wav")
    write(filename, SAMPLE_RATE, (stereo*32767).astype(np.int16))
    txt.text = f"{label} played!\nFile saved as:\n{filename}\nPress SPACE for next."
    txt.draw(); win.flip()
    event.waitKeys(keyList=['space'])

# 1. Dichotic Pitch
def make_dichotic_pitch(freq, bw):
    wn = np.random.normal(size=N_SAMPLES)
    F = fft(wn)
    f_axis = fftfreq(N_SAMPLES, 1/SAMPLE_RATE)
    band = ((abs(f_axis) > freq-bw) & (abs(f_axis) < freq+bw))
    F[band] *= -1  # phase flip
    flipped = np.real(ifft(F))
    m = max(np.max(abs(wn)), np.max(abs(flipped)))
    wn /= m; flipped /= m
    stereo = np.column_stack((wn, flipped)).astype(np.float32)
    return stereo

# 2. Huggins Pitch
def make_huggins_pitch(freq, bw):
    wn = np.random.normal(size=N_SAMPLES)
    F = fft(wn)
    f_axis = fftfreq(N_SAMPLES, 1/SAMPLE_RATE)
    band = ((abs(f_axis) > freq-bw) & (abs(f_axis) < freq+bw))
    phase_shift = np.exp(1j * np.pi * band)  # pi shift
    right = np.real(ifft(F * phase_shift))
    m = max(np.max(abs(wn)), np.max(abs(right)))
    wn /= m; right /= m
    stereo = np.column_stack((wn, right)).astype(np.float32)
    return stereo

# 3. Binaural Edge Pitch (basic version)
def make_binaural_edge(edge_freq):
    t = np.linspace(0, DURATION, N_SAMPLES, endpoint=False)
    left = np.random.normal(size=N_SAMPLES)
    right = left.copy()
    idx = int(N_SAMPLES * edge_freq / SAMPLE_RATE)
    right[idx:] *= -1  # phase inversion above edge
    m = max(np.max(abs(left)), np.max(abs(right)))
    left /= m; right /= m
    stereo = np.column_stack((left, right)).astype(np.float32)
    return stereo

# ---- Render, Play, and Save the three stimuli ----
dichotic = make_dichotic_pitch(FREQ, BANDWIDTH)
play_stim_and_save(dichotic, "Dichotic Pitch", DP_DIR)

huggins = make_huggins_pitch(FREQ, BANDWIDTH)
play_stim_and_save(huggins, "Huggins Pitch", HP_DIR)

edge = make_binaural_edge(EDGE_FREQ)
play_stim_and_save(edge, "Binaural Edge Pitch", BE_DIR)

txt.text = "All stimuli done and saved!\nPress SPACE to exit."
txt.draw(); win.flip()
event.waitKeys(keyList=['space'])
win.close()
core.quit()
