import numpy as np
import sounddevice as sd
import scipy.signal
from scipy.io.wavfile import write as wavwrite

# --- Utility Functions ---

def bandpass_filter(signal, fs, lowcut, highcut):
    sos = scipy.signal.butter(6, [lowcut, highcut], btype='band', fs=fs, output='sos')
    return scipy.signal.sosfilt(sos, signal)

def cosine_window(signal, window_length):
    w = np.hanning(2 * window_length)
    fade_in = w[:window_length]
    fade_out = w[window_length:]
    signal[:window_length] *= fade_in
    signal[-window_length:] *= fade_out
    return signal

def normalize(signal):
    return signal / np.max(np.abs(signal))

def create_dichotic_pitch(sbr, peak, width, tsSig, tsBack, fs, duration):
    t = np.arange(0, int(fs * duration)) / fs
    noise_amplitude = 1.0  # Try values from 0.5 to 1.5 depending on desired loudness
    noiseL = np.random.normal(0, noise_amplitude, len(t))
    noiseR = np.random.normal(0, noise_amplitude, len(t))


    def gen_tone_component(peak_freqs, delay_ms):
        if np.isscalar(peak_freqs):
            peak_freqs = [peak_freqs]
        comp = np.zeros_like(t)
        for freq in peak_freqs:
            comp += np.sin(2 * np.pi * freq * t)
        comp /= len(peak_freqs)
        shift_samples = int(delay_ms * 1e-3 * fs)
        return np.roll(comp, shift_samples)

    signalL = np.zeros_like(t)
    signalR = np.zeros_like(t)

    if sbr > 0:
        signalL = gen_tone_component(peak, tsSig)
        signalR = gen_tone_component(peak, tsBack)
        signalL *= sbr
        signalR *= sbr

    mixedL = noiseL + signalL
    mixedR = noiseR + signalR

    stereo = np.column_stack((mixedL, mixedR))
    return stereo


# --- Constants ---
fs = 11025
fMax = 1200
noteDuration = 10.0
coswin_len = int(fs / 100)  # 10 ms

# --- Demo 1: Simple Dichotic Pitch ---
dp = create_dichotic_pitch(
    sbr=1, peak=500, width=500/20, tsSig=-0.6, tsBack=0.0,
    fs=fs, duration=noteDuration
)
dp = bandpass_filter(dp, fs, 20, fMax)
dp[:, 0] = cosine_window(dp[:, 0], coswin_len)
dp[:, 1] = cosine_window(dp[:, 1], coswin_len)
dp = normalize(dp)

print("Playing Simple DP...")
sd.play(dp, fs)
sd.wait()
wavwrite("dp.wav", fs, (dp * 32767).astype(np.int16))

# --- Demo 2: SBR Sequence ---
sbrs = 10 ** np.arange(np.log10(8), np.log10(0.5) - 0.01, -0.3)
print("SBR levels:", sbrs.round(2))
sequence = []
for sbr in sbrs:
    dp1 = create_dichotic_pitch(sbr, [300, 600, 900], width=[15, 30, 45],
                                tsSig=0.6, tsBack=-0.6, fs=fs, duration=0.3)
    blank = create_dichotic_pitch(0, [300, 600, 900], width=[15, 30, 45],
                                  tsSig=0.6, tsBack=-0.6, fs=fs, duration=0.3)
    sequence.append(dp1)
    sequence.append(blank)
dp2 = np.vstack(sequence)
dp2 = bandpass_filter(dp2, fs, 20, fMax)
dp2[:, 0] = cosine_window(dp2[:, 0], coswin_len)
dp2[:, 1] = cosine_window(dp2[:, 1], coswin_len)
dp2 = normalize(dp2)

print("Playing SBR sequence...")
sd.play(dp2, fs)
sd.wait()
wavwrite("dpSBR.wav", fs, (dp2 * 32767).astype(np.int16))

# --- Demo 3: Dichotic Pitch Melody ---
melody = [
    {'peak': 0, 'tsSig': 0.6, 'duration': 0.1},
    {'peak': [300, 600, 900], 'tsSig': 0.6, 'duration': 0.3},
    {'peak': 0, 'tsSig': 0.6, 'duration': 0.1},
    {'peak': [400, 800], 'tsSig': 0.6, 'duration': 0.3},
    {'peak': 0, 'tsSig': 0.6, 'duration': 0.1},
    {'peak': [300, 600, 900], 'tsSig': 0.6, 'duration': 0.3},
    {'peak': 0, 'tsSig': 0.6, 'duration': 0.1},
    {'peak': [200, 400, 600, 800], 'tsSig': 0.6, 'duration': 0.3},
    {'peak': 0, 'tsSig': 0.6, 'duration': 0.1},
]

melody_wave = []
for note in melody:
    peak = note['peak']
    if peak == 0:
        peak = [300]  # dummy freq, will be zeroed out
        sbr = 0
    else:
        sbr = 1
    width = np.array(peak) / 20
    dp_note = create_dichotic_pitch(sbr, peak, width, note['tsSig'], -0.6, fs, note['duration'])
    melody_wave.append(dp_note)

dp3 = np.vstack(melody_wave)
dp3 = bandpass_filter(dp3, fs, 20, fMax)
dp3[:, 0] = cosine_window(dp3[:, 0], coswin_len)
dp3[:, 1] = cosine_window(dp3[:, 1], coswin_len)
dp3 = normalize(dp3)

print("Playing DP melody...")
sd.play(dp3, fs)
sd.wait()
wavwrite("dpMelody.wav", fs, (dp3 * 32767).astype(np.int16))
