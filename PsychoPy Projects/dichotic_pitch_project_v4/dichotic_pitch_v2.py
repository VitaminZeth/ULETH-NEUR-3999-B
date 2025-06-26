#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dichotic-Pitch EEG Experiment
– Pre-session screening
– Headphone check (loops until passed)
– Pure-tone hearing threshold test
– Practice block
– Main experiment (compensated & uncompensated phases)
"""

# 1) FORCE SOUNDDEVICE BACKEND (must come before any psychopy.sound import)
from psychopy import prefs
prefs.hardware['audioLib'] = ['sounddevice']
    
# 2) IMPORTS
from psychopy import sound, core, event, gui, visual
import numpy as np
from scipy.fft import fft, ifft, fftfreq
from scipy.io.wavfile import write
import pandas as pd
import random, os
from datetime import datetime

# --- Experiment constants ---
SAMPLE_RATE = 44100
DURATION    = 1.0
N_SAMPLES   = int(SAMPLE_RATE * DURATION)
BANDWIDTH   = 20
FREQUENCIES = [250, 400, 600, 800, 1000]
CONTROL_FREQ= 500
NUM_BLOCKS  = 2
RETSPL      = {250:26.5,500:13.5,1000:7.5,2000:9.0,4000:9.0,8000:13.5}
CALIB_REF   = 94.0   # dB SPL for full-scale
SL_TARGET   = 40     # Sensation level above threshold

# --- Utility functions ---
def hl_to_amplitude(freq, hl_dbhl):
    """Convert hearing level (dB HL) to normalized amplitude (0.0–1.0)."""
    spl = hl_dbhl + RETSPL[freq]
    return 10 ** ((spl - CALIB_REF) / 20)

def create_dichotic_pitch(freq, bw=BANDWIDTH, flip='right'):
    """Generate a phase-flipped band of white noise for dichotic pitch."""
    wn = np.random.normal(size=N_SAMPLES)
    F  = fft(wn)
    f_axis = fftfreq(N_SAMPLES, 1/SAMPLE_RATE)
    # flip π phase in the target band
    band = ((abs(f_axis)>freq-bw) & (abs(f_axis)<freq+bw))
    F[band] *= -1
    flipped = np.real(ifft(F))
    # normalize
    m = max(np.max(abs(wn)), np.max(abs(flipped)))
    wn /= m; flipped /= m
    # stereo mix
    if flip=='left':
        stereo = np.column_stack((flipped, wn))
    elif flip=='right':
        stereo = np.column_stack((wn, flipped))
    else:
        stereo = np.column_stack((wn, wn))
    return stereo.astype(np.float32)

def save_stim_and_analysis(stereo, freq, flip, comp_flag, outdir):
    """Write WAV + FFT CSV analysis."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_path = os.path.join(outdir, f"stim_{freq}Hz_{flip}_{ts}.wav")
    write(wav_path, SAMPLE_RATE, (stereo*32767).astype(np.int16))
    # analyze flipped channel
    ch = 1 if flip=='right' else 0
    F  = fft(stereo[:,ch])
    mags = 20*np.log10(np.abs(F)+1e-12)
    phs  = np.angle(F)
    freqs= fftfreq(N_SAMPLES,1/SAMPLE_RATE)
    df = pd.DataFrame({
        'Frequency': freqs[:N_SAMPLES//2],
        'Magnitude_dB': mags[:N_SAMPLES//2],
        'Phase_rad': phs[:N_SAMPLES//2],
        'FlippedBand': band[:N_SAMPLES//2],
        'Channel': flip,
        'Compensated': comp_flag
    })
    df.to_csv(wav_path.replace('.wav','.csv'), index=False)

# 3) PARTICIPANT SCREENING
scr = {
    'Participant ID:':'',
    'Session #:'      :'001',
    'Hearing issues?':'n',
    'Neuro history?':'n',
    'Implants?':'n',
    'Medications?':'n',
    'Pregnant?':'n'
}
dlg = gui.DlgFromDict(scr, title="Pre-session Screening")
# PsychoPy’s DlgFromDict uses dlg.OK on some versions, dlg.ok on others:
if not getattr(dlg, 'OK', getattr(dlg, 'ok', False)):
    core.quit()
pid, session = scr['Participant ID:'], scr['Session #:']
# unpack
hearing_issues = scr['Hearing issues?']
neuro_history  = scr['Neuro history?']
implants       = scr['Implants?']
medications    = scr['Medications?']
pregnant       = scr['Pregnant?']

# 4) DIRECTORIES & FILE
out_dir = f"data/{pid}_{session}"
os.makedirs(out_dir, exist_ok=True)
stim_dir = os.path.join(out_dir,'stimuli')
os.makedirs(stim_dir, exist_ok=True)
# CSV header
csv_path = os.path.join(out_dir,f"{pid}_{session}.csv")
header = [
 'block','trial','phase','type','freq','heard','conf','rt',
 'ear','pitch','correct','flipped','ear_correct',
 'L250','L500','L1000','L2000','L4000','L8000',
 'R250','R500','R1000','R2000','R4000','R8000',
 'hear_issues','neuro','implants','meds','preg','eeg_imp','vol_ok','break_ok','compensated'
]
data_f = open(csv_path,'w')
data_f.write(','.join(header)+'\n')

# 5) WINDOW SETUP
win = visual.Window(fullscr=False, color='black')
txt = visual.TextStim(win, color='white', height=0.08, wrapWidth=1.5)
fix = visual.TextStim(win, text='+', color='white', height=0.1)

# 6) EEG IMPEDANCE & VOLUME CAL
imp = {'Electrodes <5kΩ & stable?':'y'}
dlg2 = gui.DlgFromDict(imp, title='EEG Impedance')
if not getattr(dlg2,'OK',getattr(dlg2,'ok',False)): core.quit()
eeg_ok = imp[list(imp.keys())[0]]
txt.text = "Adjust volume comfortably, then press SPACE"
txt.draw(); win.flip()
event.waitKeys(keyList=['space'])

# 7) HEADPHONE CHECK (loops until pass)
while True:
    txt.text = "HEADPHONE CHECK\n\nPress SPACE to begin or ESC to quit"
    txt.draw(); win.flip()
    keys = event.waitKeys(keyList=['space','escape'])
    if 'escape' in keys: core.quit()
    passed = True
    for _ in range(3):
        side = random.choice(['left','right'])
        tone = np.zeros((N_SAMPLES,2))
        t = np.linspace(0,DURATION,N_SAMPLES)
        tone[:,0 if side=='left' else 1] = np.sin(2*np.pi*1000*t)
        s = sound.Sound(value=tone, sampleRate=SAMPLE_RATE, stereo=True)
        s.play(); core.wait(DURATION)
        txt.text = "Which ear? (LEFT/RIGHT) or ESC"
        txt.draw(); win.flip()
        k = event.waitKeys(keyList=['left','right','escape'])[0]
        if k=='escape': core.quit()
        if k!=side: passed=False
    if passed: break
    txt.text = "Headphone check FAILED. Retrying..."
    txt.draw(); win.flip(); core.wait(1.5)

# 8) PURE-TONE HEARING TEST
txt.text = "PURE-TONE TEST\nPress SPACE to begin or ESC to quit"
txt.draw(); win.flip()
keys = event.waitKeys(keyList=['space','escape'])
if 'escape' in keys: core.quit()
pure_freqs = [250,500,1000,2000,4000,8000]
thresholds = {'Left':{}, 'Right':{}}
for ear in ['Left','Right']:
    ch = 0 if ear=='Left' else 1
    txt.text = f"Pure-tone test: {ear} ear. Press SPACE"
    txt.draw(); win.flip()
    event.waitKeys(keyList=['space'])
    for f in pure_freqs:
        lvl = 50; heard_once=False
        while True:
            amp = hl_to_amplitude(f, lvl)
            t = np.linspace(0,DURATION,N_SAMPLES)
            tone = np.zeros((N_SAMPLES,2))
            tone[:,ch] = amp * np.sin(2*np.pi*f*t)
            sound.Sound(value=tone, sampleRate=SAMPLE_RATE, stereo=True).play()
            core.wait(DURATION)
            txt.text = f"Hear {f}Hz at {lvl} dB? (y/n)"
            txt.draw(); win.flip()
            k = event.waitKeys(keyList=['y','n'])[0]
            if k=='y':
                heard_once=True
                lvl -= 10
            else:
                if heard_once:
                    lvl += 5
                    break
                lvl += 5
            lvl = int(np.clip(lvl,0,100))
        thresholds[ear][f] = lvl

# Record thresholds in CSV (zero-block screening row)
row0 = ['0','0','Screen','NA','NA','NA','NA','NA','NA','NA','NA','NA','NA']
# append L and R thresholds
row0 += [ thresholds['Left'][f] for f in pure_freqs ]
row0 += [ thresholds['Right'][f] for f in pure_freqs ]
row0 += [ hearing_issues, neuro_history, implants, medications, pregnant, eeg_ok, 'yes', 'NA' ]
data_f.write(','.join(map(str,row0))+'\n')
data_f.flush()

# 9) PRACTICE BLOCK
txt.text = "PRACTICE BLOCK\nPress SPACE to continue"
txt.draw(); win.flip(); event.waitKeys(keyList=['space'])
for f in [400,800]:
    stim = create_dichotic_pitch(f, bw=50, flip='right')
    sound.Sound(value=stim, sampleRate=SAMPLE_RATE, stereo=True).play()
    core.wait(DURATION)
    txt.text = "Did you hear a pitch? (y/n)"
    txt.draw(); win.flip()
    event.waitKeys(keyList=['y','n'])
    core.wait(0.5)

# 10) MAIN EXPERIMENT FUNCTION
def run_phase(compensated):
    phase = "Compensated" if compensated else "Uncompensated"
    txt.text = f"Starting {phase} phase\nPress SPACE"
    txt.draw(); win.flip(); event.waitKeys(keyList=['space'])
    for blk in range(1,NUM_BLOCKS+1):
        tests = random.sample([f for f in FREQUENCIES if f!=CONTROL_FREQ],3)
        seq   = [CONTROL_FREQ] + tests + [CONTROL_FREQ]
        for tr, f in enumerate(seq,1):
            typ  = 'Control' if f==CONTROL_FREQ else 'Test'
            flip = random.choice(['left','right']) if typ=='Test' else 'none'
            stim = create_dichotic_pitch(f, flip=flip)
            # apply individual compensation
            if compensated and typ=='Test':
                lvl = thresholds[flip.capitalize()][f] + SL_TARGET
                amp = hl_to_amplitude(f, lvl)
                stim *= amp
            # save only Test stimuli
            if typ=='Test':
                save_stim_and_analysis(stim, f, flip, compensated, stim_dir)
            # trial sequence
            fix.draw(); win.flip(); core.wait(1.0)
            sound.Sound(value=stim, sampleRate=SAMPLE_RATE, stereo=True).play()
            core.wait(DURATION)
            txt.text = "Hear a pitch? (y/n)"
            txt.draw(); win.flip()
            key, rt = event.waitKeys(keyList=['y','n'], timeStamped=core.Clock())[0]
            txt.text = "Confidence? (1–5)"
            txt.draw(); win.flip()
            conf = event.waitKeys(keyList=[str(i) for i in range(1,6)])[0]
            if key=='y':
                txt.text = "Which ear? (left/right)"; txt.draw(); win.flip()
                ear = event.waitKeys(keyList=['left','right'])[0]
                txt.text = "High or Low? (h/l)"; txt.draw(); win.flip()
                pt  = event.waitKeys(keyList=['h','l'])[0]
            else:
                ear, pt = 'NA','NA'
            corr     = 'yes' if (typ=='Test' and key=='y') or (typ=='Control' and key=='n') else 'no'
            ear_corr = 'yes' if flip==ear else 'no' if flip in ['left','right'] else 'NA'
            # write row
            row = [
              blk, tr, phase, typ, f, key, conf, f"{rt:.3f}", ear,
              'high' if pt=='h' else 'low' if pt=='l' else 'NA',
              corr, flip, ear_corr
            ]
            # pad for thresholds, append metadata
            row += ['']*12
            row += [hearing_issues, neuro_history, implants, medications, pregnant, eeg_ok, '', compensated]
            data_f.write(','.join(map(str,row))+'\n')
            data_f.flush()
            core.wait(0.5)
        # break screen
        if blk<NUM_BLOCKS:
            txt.text = "Short break – SPACE to continue or n to end"
            txt.draw(); win.flip()
            k = event.waitKeys(keyList=['space','n'])[0]
            br = 'yes' if k=='space' else 'no'
            data_f.write(f"{blk},0,{phase},Break,,,,,,,,,,,,,,,,{br},{compensated}\n")
            data_f.flush()

# 11) RUN EXPERIMENT
run_phase(compensated=True)
run_phase(compensated=False)

# 12) CLEANUP
data_f.close()
txt.text = "Experiment complete. Thank you!"
txt.draw(); win.flip()
core.wait(2.0)
win.close()
core.quit()
