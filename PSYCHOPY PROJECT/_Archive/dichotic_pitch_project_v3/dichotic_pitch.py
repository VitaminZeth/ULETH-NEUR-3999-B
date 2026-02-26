# === Dichotic Pitch EEG Experiment ===
from psychopy import prefs
prefs.hardware['audioLib'] = ['PTB']  # Use PTB for EEG timing accuracy

from psychopy import sound, core, event, gui, data, logging, visual
from egi.simple import Netstation
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.fftpack import ifft
from scipy.io.wavfile import write
import pandas as pd
import random
import os
from datetime import datetime

# === GUI ===
exp_info = {'Participant': '', 'Session': '001'}
dlg = gui.DlgFromDict(exp_info, title='Dichotic Pitch EEG')
if not dlg.OK:
    core.quit()

filename = f"data/{exp_info['Participant']}_{exp_info['Session']}"
os.makedirs("data", exist_ok=True)
os.makedirs("stimuli", exist_ok=True)
data_file = open(filename + ".csv", "w")
data_file.write("block,trial,type,frequency,event_code,heard,confidence,response_time\n")

# === EEG Connection ===
netstation = Netstation("10.10.10.42", 55513)
netstation.connect()
netstation.beginSession()
netstation.sync()

# === Visuals ===
win = visual.Window(fullscr=False, color='black')
text_stim = visual.TextStim(win, text='', color='white', height=0.07)

# === Parameters ===
sample_rate = 48000
n_samples = int(sample_rate * 1.0)
duration = 1.0
bandwidth = 20
control_freq = 500
frequencies = [250, 400, 600, 800, 1000]
event_codes = {250: 'F250', 400: 'F400', 600: 'F600', 800: 'F800', 1000: 'F1000', 500: 'CTRL'}
num_blocks = 2

# === Stimulus Generator ===
def create_dichotic_pitch(freq):
    white_noise = np.random.normal(0, 1, n_samples)
    freq_noise = fft(white_noise)
    freqs = fftfreq(n_samples, d=1/sample_rate)
    freq_noise_shifted = freq_noise.copy()
    flipped_band = []
    for i, f in enumerate(freqs):
        if freq - bandwidth < abs(f) < freq + bandwidth:
            freq_noise_shifted[i] *= np.exp(1j * np.pi)
            flipped_band.append(f)
    right_noise = np.real(ifft(freq_noise_shifted))
    max_val = max(np.max(np.abs(white_noise)), np.max(np.abs(right_noise)))
    left = white_noise / max_val
    right = right_noise / max_val
    stereo = np.vstack((left, right)).T.astype(np.float32)
    return stereo, freqs, right_noise, flipped_band

# === WAV + FFT Analysis ===
def save_stimulus_and_analysis(stereo, freqs, right, flipped_band, freq):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_path = f"stimuli/stimulus_{freq}Hz_{timestamp}.wav"
    write(wav_path, sample_rate, (stereo * 32767).astype(np.int16))

    fft_right = fft(right)
    magnitude = 20 * np.log10(np.abs(fft_right) + 1e-12)
    phase = np.angle(fft_right)
    df = pd.DataFrame({
        'Frequency (Hz)': freqs[:n_samples // 2],
        'Loudness (dBFS)': magnitude[:n_samples // 2],
        'Phase (radians)': phase[:n_samples // 2]
    })
    df['Phase-Shifted'] = df['Frequency (Hz)'].apply(lambda f: 'YES' if freq - bandwidth < abs(f) < freq + bandwidth else '')
    csv_path = f"stimuli/fft_analysis_{freq}Hz_{timestamp}.csv"
    df.to_csv(csv_path, index=False)

# === Block/Trial Logic ===
for block_num in range(1, num_blocks + 1):
    rand_freq = random.choice(frequencies)
    block_trials = [control_freq] + [rand_freq]*3 + [control_freq]

    for trial_num, freq in enumerate(block_trials):
        trial_type = 'Control' if freq == control_freq else 'Test'
        stim_array, freqs, right, flipped_band = create_dichotic_pitch(freq)
        stim = sound.Sound(value=stim_array, sampleRate=sample_rate, stereo=True)

        if trial_type == 'Test':
            save_stimulus_and_analysis(stim_array, freqs, right, flipped_band, freq)

        # Show trial instructions
        text_stim.text = f"Block {block_num} - Trial {trial_num+1}\nListen carefully."
        text_stim.draw()
        win.flip()
        core.wait(1.0)

        # Pre-stim EEG baseline
        win.flip()
        core.wait(1.0)

        # Send EEG trigger
        netstation.send_event(event=event_codes[freq], label=event_codes[freq])

        # Play audio
        stim.play()
        core.wait(duration)

        # Response collection
        text_stim.text = "Did you hear a pitch? (y/n)"
        text_stim.draw()
        win.flip()
        clock = core.Clock()
        keys = event.waitKeys(keyList=['y', 'n'], timeStamped=clock)
        pitch_resp, rt = keys[0]

        text_stim.text = "Confidence? (1-5)"
        text_stim.draw()
        win.flip()
        conf_keys = event.waitKeys(keyList=['1','2','3','4','5'], timeStamped=clock)
        conf_resp, conf_rt = conf_keys[0]

        data_file.write(f"{block_num},{trial_num+1},{trial_type},{freq},{event_codes[freq]},{pitch_resp},{conf_resp},{rt:.3f}\n")
        data_file.flush()
        core.wait(0.5)

    # Break between blocks
    if block_num < num_blocks:
        text_stim.text = "Take a short break.\nPress space to continue."
        text_stim.draw()
        win.flip()
        event.waitKeys(keyList=['space'])

# === Cleanup ===
data_file.close()
netstation.endSession()
netstation.disconnect()
text_stim.text = "Experiment Complete. Thank you!"
text_stim.draw()
win.flip()
core.wait(3.0)
win.close()
core.quit()