
from psychopy import prefs
prefs.hardware['audioLib'] = ['sounddevice']

from psychopy import sound, core, event, gui, visual
import numpy as np
from scipy.fft import fft, ifft, fftfreq
from scipy.io.wavfile import write
import pandas as pd
import random
from datetime import datetime
import os

# === GUI ===
exp_info = {'Participant': '', 'Session': '001'}
dlg = gui.DlgFromDict(exp_info, title='Dichotic Pitch EEG (Offline Pilot)')
if not dlg.OK:
    core.quit()

filename = f"data/{exp_info['Participant']}_{exp_info['Session']}"
os.makedirs("data", exist_ok=True)
os.makedirs("stimuli", exist_ok=True)
data_file = open(filename + ".csv", "w")
data_file.write("block,trial,type,frequency,heard,confidence,response_time,ear_perceived,pitch_type,correct,channel_flipped,ear_correct\n")

# === VISUAL ===
win = visual.Window(fullscr=False, color='black')
text_stim = visual.TextStim(win, text='', color='white', height=0.07)
fixation = visual.TextStim(win, text='+', color='white', height=0.1)

# === PARAMETERS ===
sample_rate = 44100
duration = 1.0
n_samples = int(sample_rate * duration)
bandwidth = 20
frequencies = [250, 400, 600, 800, 1000]
control_freq = 500
num_blocks = 2

# === FUNCTIONS ===
def create_dichotic_pitch(freq, wide=False, flip_channel='right'):
    bw = 50 if wide else bandwidth
    white_noise = np.random.normal(0, 1, n_samples)
    freq_noise = fft(white_noise)
    freqs = fftfreq(n_samples, d=1/sample_rate)
    freq_noise_shifted = freq_noise.copy()
    flipped_band = []
    for i, f in enumerate(freqs):
        if freq - bw < abs(f) < freq + bw:
            freq_noise_shifted[i] *= np.exp(1j * np.pi)
            flipped_band.append(f)
    flipped_noise = np.real(ifft(freq_noise_shifted))
    max_val = max(np.max(np.abs(white_noise)), np.max(np.abs(flipped_noise)))
    base = white_noise / max_val
    flipped = flipped_noise / max_val
    if flip_channel == 'left':
        stereo = np.vstack((flipped, base)).T
    elif flip_channel == 'right':
        stereo = np.vstack((base, flipped)).T
    else:
        stereo = np.vstack((base, base)).T
    return stereo.astype(np.float32), freqs, flipped_noise, flipped_band

def save_stimulus_and_analysis(stereo, freqs, flipped_noise, flipped_band, freq):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_path = f"stimuli/stimulus_{freq}Hz_{timestamp}.wav"
    write(wav_path, sample_rate, (stereo * 32767).astype(np.int16))

    fft_right = fft(flipped_noise)
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

# === INSTRUCTIONS ===
text_stim.text = "Welcome! This experiment tests pitch perception using headphones.\n\nPress space to begin."
text_stim.draw()
win.flip()
event.waitKeys(keyList=['space'])

# === EXPERIMENT BLOCKS ===
for block_num in range(1, num_blocks + 1):
    rand_freq = random.choice(frequencies)
    block_trials = [control_freq] + [rand_freq]*3 + [control_freq]

    for trial_num, freq in enumerate(block_trials):
        trial_type = 'Control' if freq == control_freq else 'Test'
        pitch_label = 'high' if freq >= 600 else 'low'
        flip_channel = random.choice(['left', 'right']) if trial_type == 'Test' else 'none'

        stim_array, freqs, flipped_noise, flipped_band = create_dichotic_pitch(freq, flip_channel=flip_channel)
        stim = sound.Sound(value=stim_array, sampleRate=sample_rate, stereo=True)

        if trial_type == 'Test':
            save_stimulus_and_analysis(stim_array, freqs, flipped_noise, flipped_band, freq)

        fixation.draw()
        win.flip()
        core.wait(1.0)

        stim.play()
        core.wait(duration)

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
        conf_resp, _ = conf_keys[0]

        text_stim.text = "Which ear? (left/right)"
        text_stim.draw()
        win.flip()
        ear_resp = event.waitKeys(keyList=['left', 'right'])[0]

        text_stim.text = "Was the pitch high or low? (h/l)"
        text_stim.draw()
        win.flip()
        pitch_type_resp = event.waitKeys(keyList=['h', 'l'])[0]

        correct = 'yes' if (trial_type == 'Control' and pitch_resp == 'n') or (trial_type == 'Test' and pitch_resp == 'y') else 'no'
        ear_correct = 'NA'
        if trial_type == 'Test':
            ear_correct = 'yes' if flip_channel == ear_resp else 'no'

        data_file.write(f"{block_num},{trial_num+1},{trial_type},{freq},{pitch_resp},{conf_resp},{rt:.3f},"
                        f"{ear_resp},{'high' if pitch_type_resp=='h' else 'low'},{correct},{flip_channel},{ear_correct}\n")
        data_file.flush()
        core.wait(0.5)

    if block_num < num_blocks:
        text_stim.text = "Take a short break.\nPress space to continue."
        text_stim.draw()
        win.flip()
        event.waitKeys(keyList=['space'])

data_file.close()
text_stim.text = "Experiment Complete. Thank you!"
text_stim.draw()
win.flip()
core.wait(3.0)
win.close()
core.quit()
