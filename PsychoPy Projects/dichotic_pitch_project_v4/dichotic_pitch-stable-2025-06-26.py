from psychopy import prefs
prefs.hardware['audioLib'] = ['sounddevice']  # use sounddevice backend

from psychopy import sound, core, event, gui, visual
import numpy as np
from scipy.fft import fft, ifft, fftfreq
from scipy.io.wavfile import write
import pandas as pd
import random
from datetime import datetime
import os
import matplotlib.pyplot as plt

# === Calibration for Pure-Tone Conversion ===
RETSPL = {250:26.5, 500:13.5, 1000:7.5, 2000:9.0, 4000:9.0, 8000:13.5}
CALIB_SPL_REF = 94.0  # dB SPL at full-scale
SENSATION_LEVEL = 40  # desired SL above threshold for compensation

def hl_to_amplitude(freq, hl):
    """Convert hearing level (dB HL) to digital amplitude (0.0–1.0)."""
    threshold_spl = hl + RETSPL[freq]
    return 10 ** ((threshold_spl - CALIB_SPL_REF) / 20)

# === Dichotic Pitch Stimulus Generation ===
def create_dichotic_pitch(freq, wide=False, flip_channel='right'):
    bw = 50 if wide else bandwidth
    white_noise = np.random.normal(0, 1, n_samples)
    freq_noise = fft(white_noise)
    freqs = fftfreq(n_samples, d=1/sample_rate)
    freq_noise_shifted = freq_noise.copy()
    for i, f in enumerate(freqs):
        if freq - bw < abs(f) < freq + bw:
            freq_noise_shifted[i] *= np.exp(1j * np.pi)
    flipped_noise = np.real(ifft(freq_noise_shifted))
    max_val = max(np.max(np.abs(white_noise)), np.max(np.abs(flipped_noise)))
    white_norm = white_noise / max_val
    flipped_norm = flipped_noise / max_val
    if flip_channel == 'left':
        stereo = np.column_stack((flipped_norm, white_norm))
    elif flip_channel == 'right':
        stereo = np.column_stack((white_norm, flipped_norm))
    else:
        stereo = np.column_stack((white_norm, white_norm))
    return stereo.astype(np.float32), freqs, flipped_noise

# === Stimulus Analysis ===
def save_stimulus_and_analysis(stereo, freqs, flipped_noise, freq, flip_channel, comp_flag):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    write(f"{stim_dir}/stimulus_{freq}Hz_{timestamp}.wav", sample_rate, (stereo * 32767).astype(np.int16))
    mag = 20 * np.log10(np.abs(fft(flipped_noise)) + 1e-12)
    phi = np.angle(fft(flipped_noise))
    df = pd.DataFrame({
        'Frequency (Hz)': freqs[:n_samples//2],
        'Loudness (dBFS)': mag[:n_samples//2],
        'Phase (rad)': phi[:n_samples//2],
        'Channel Flipped': flip_channel,
        'Compensated': comp_flag
    })
    df.to_csv(f"{stim_dir}/fft_{freq}Hz_{timestamp}.csv", index=False)

# === Participant Screening ===
# collect participant metadata via a dict-based dialog
scr_dict = {
    'Participant ID:':'',
    'Session #:':'001',
    'Hearing issues? (y/n)':'n',
    'Neurological history? (y/n)':'n',
    'Metal implants? (y/n)':'n',
    'Medications? (y/n)':'n',
    'Pregnant? (y/n)':'n'
}
screen = gui.DlgFromDict(scr_dict, title='Pre-session Screening')
if not screen.OK:
    core.quit()
# unpack metadata
pid = scr_dict['Participant ID:']
session = scr_dict['Session #:']
hearing_issues = scr_dict['Hearing issues? (y/n)']
neuro_history = scr_dict['Neurological history? (y/n)']
implants = scr_dict['Metal implants? (y/n)']
medications = scr_dict['Medications? (y/n)']
pregnant = scr_dict['Pregnant? (y/n)']

# === Experiment Parameters & Setup === & Setup ===
sample_rate = 44100
duration = 1.0
n_samples = int(sample_rate * duration)
bandwidth = 20
frequencies = [250, 400, 600, 800, 1000]
control_freq = 500
num_blocks = 2
output_dir = f"data/{pid}_{session}"
os.makedirs(output_dir, exist_ok=True)
stim_dir = os.path.join(output_dir, 'stimuli')
os.makedirs(stim_dir, exist_ok=True)

# Expanded header with compensation flag
data_file = open(os.path.join(output_dir, f"{pid}_{session}.csv"), 'w')
header = [
    'block','trial','phase','type','frequency','heard','confidence','response_time',
    'ear_perceived','pitch_type','correct','channel_flipped','ear_correct',
    'L250','L500','L1000','L2000','L4000','L8000',
    'R250','R500','R1000','R2000','R4000','R8000',
    'hearing_issues','neuro_history','implants','medications','pregnant',
    'eeg_imp','vol_cal','break_ok','compensated'
]
data_file.write(','.join(header) + '\n')

# === Visual Elements ===
win = visual.Window(fullscr=False, color='black')
text_stim = visual.TextStim(win, text='', color='white', height=0.08)
fixation = visual.TextStim(win, text='+', color='white', height=0.1)

# === EEG Impedance & Volume Calibration ===
# use dict-based dialog to get explicit OK
imp_dict = {'Electrodes <5kΩ and stable? (y/n)':'y'}
dlg_imp = gui.DlgFromDict(imp_dict, title='EEG Impedance Check')
if not dlg_imp.OK:
    core.quit()
eeg_imp = imp_dict['Electrodes <5kΩ and stable? (y/n)']

# Volume calibration prompt
text_stim.text = "Adjust volume comfortably, then press SPACE."
text_stim.draw(); win.flip(); event.waitKeys(['space'])
vol_cal = 'yes'

# === Headphone Check ===
text_stim.text = "HEADPHONE CHECK:\nPress SPACE to start."
text_stim.draw(); win.flip(); event.waitKeys(['space'])
pass_hp = True
for _ in range(3):
    side = random.choice(['left','right'])
    t = np.linspace(0, duration, n_samples)
    tone = np.zeros((n_samples, 2)); tone[:,0 if side=='left' else 1] = np.sin(2*np.pi*1000*t)
    sound.Sound(value=tone, sampleRate=sample_rate, stereo=True).play(); core.wait(duration)
    text_stim.text = "Which ear? (left/right)"; text_stim.draw(); win.flip()
    if event.waitKeys(['left','right'])[0] != side: pass_hp = False
if not pass_hp:
    text_stim.text = "Headphone check failed."; text_stim.draw(); win.flip(); core.wait(3); core.quit()

# === Pure-Tone Hearing Test ===
def run_pure_tone_test():
    freqs = [250,500,1000,2000,4000,8000]
    thr = {'Left':{}, 'Right':{}}
    for ear in ['Left','Right']:
        ch = 0 if ear=='Left' else 1
        text_stim.text = f"Pure-Tone Test: {ear} Ear\nPress SPACE to begin"
        text_stim.draw(); win.flip(); event.waitKeys(['space'])
        for f in freqs:
            lvl = 50; heard_once=False
            while True:
                amp = hl_to_amplitude(f, lvl)
                t = np.linspace(0, duration, n_samples)
                tone = np.zeros((n_samples,2)); tone[:,ch] = amp*np.sin(2*np.pi*f*t)
                sound.Sound(value=tone, sampleRate=sample_rate, stereo=True).play(); core.wait(duration)
                text_stim.text = f"Hear {f}Hz at {lvl} dB? (y/n)"; text_stim.draw(); win.flip()
                key = event.waitKeys(['y','n'])[0]
                if key=='y': heard_once=True; lvl-=10
                else:
                    if heard_once: lvl+=5; break
                    lvl+=5
                lvl = np.clip(lvl,0,100)
            thr[ear][f] = lvl
    return thr
pt = run_pure_tone_test()

# Plot and log thresholds
for ear in ['Left','Right']:
    frs = sorted(pt[ear].keys()); ths = [pt[ear][f] for f in frs]
    plt.figure(); plt.semilogx(frs, ths,'o-'); plt.gca().invert_yaxis()
    plt.title(f"{ear} Ear Audiogram"); plt.grid(True); plt.show()
data_file.write(','.join([
    '0','0','Screening','NA','NA','NA','NA','NA','NA','NA','NA','NA',
    *[str(pt['Left'][f]) for f in [250,500,1000,2000,4000,8000]],
    *[str(pt['Right'][f]) for f in [250,500,1000,2000,4000,8000]],
    hearing_issues, neuro_history, implants, medications, pregnant,
    eeg_imp, vol_cal, 'NA','NA'
]) + '\n')

def run_experiment(comp_flag):
    phase = 'Compensated' if comp_flag else 'Uncompensated'
    text_stim.text = f"Starting {phase} phase. Press SPACE"; text_stim.draw(); win.flip(); event.waitKeys(['space'])
    for blk in range(1, num_blocks+1):
        tests = random.choices([f for f in frequencies if f!=control_freq], k=3)
        trials = [control_freq] + tests + [control_freq]
        for tn,freq in enumerate(trials):
            tp = 'Control' if freq==control_freq else 'Test'
            flip = random.choice(['left','right']) if tp=='Test' else 'none'
            stim_arr, frqs, flipped = create_dichotic_pitch(freq, flip_channel=flip)
            # apply compensation
            if comp_flag and tp=='Test':
                lvl = pt[flip.capitalize()][freq] + SENSATION_LEVEL
                amp = hl_to_amplitude(freq, lvl)
                stim_arr *= amp
            if tp=='Test': save_stimulus_and_analysis(stim_arr, frqs, flipped, freq, flip, comp_flag)
            fixation.draw(); win.flip(); core.wait(1.0)
            sound.Sound(value=stim_arr, sampleRate=sample_rate, stereo=True).play(); core.wait(duration)
            text_stim.text = "Hear pitch? (y/n)"; text_stim.draw(); win.flip()
            pitch, rt = event.waitKeys(keyList=['y','n'], timeStamped=core.Clock())[0]
            text_stim.text = "Confidence? (1-5)"; text_stim.draw(); win.flip()
            conf = event.waitKeys(keyList=[str(i) for i in range(1,6)])[0]
            if pitch=='y':
                text_stim.text = "Which ear?"; text_stim.draw(); win.flip(); ear = event.waitKeys(['left','right'])[0]
                text_stim.text = "High or Low? (h/l)"; text_stim.draw(); win.flip(); pt_res = event.waitKeys(['h','l'])[0]
            else: ear, pt_res = 'NA','NA'
            corr = 'yes' if (tp=='Control' and pitch=='n') or (tp=='Test' and pitch=='y') else 'no'
            ear_corr = 'yes' if flip==ear else 'no' if flip in ['left','right'] else 'NA'
            row = [blk, tn+1, phase, tp, freq, pitch, conf, f"{rt:.3f}", ear, ('high' if pt_res=='h' else 'low' if pt_res=='l' else 'NA'), corr, flip, ear_corr] + ['']*12 + [hearing_issues, neuro_history, implants, medications, pregnant, eeg_imp, vol_cal, '', str(comp_flag)]
            data_file.write(','.join(map(str,row))+'\n'); data_file.flush(); core.wait(0.5)
        if blk<num_blocks:
            text_stim.text = "Break? SPACE to continue or N to stop."; text_stim.draw(); win.flip()
            key = event.waitKeys(['space','n'])[0]; bk = 'yes' if key=='space' else 'no'
            data_file.write(f"{blk},0,{phase},Break,,,,,,,,,,,,,,,,{bk},,{comp_flag}\n"); data_file.flush()

# run compensated then uncompensated
run_experiment(True)
run_experiment(False)

# === Cleanup ===
data_file.close(); text_stim.text = "Experiment Complete."; text_stim.draw(); win.flip(); core.wait(3)
win.close(); core.quit()
