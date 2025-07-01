#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dichotic-Pitch EEG Experiment — Revision 1.0.6
Change log:
– All event.waitKeys() calls now use keyList=… so no more Timestamp-vs-list errors  
– Pure‐tone frequencies run ascending with no repeats  
– “Both” added to main pitch‐detection responses; removed from headphone‐check  
– “T” key plays a test tone during volume calibration  
– Hitting “N” at a break quits to the goodbye screen  
– Warnings suppressed so console is clean  
– Pure-tone test now advances to next frequency once a second “no” follows any “yes”  
"""

# suppress warnings & Psychopy logging spam
import warnings
warnings.filterwarnings('ignore')
from psychopy import logging
logging.console.setLevel(logging.ERROR)

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
    spl = hl_dbhl + RETSPL.get(freq, 0)
    return 10 ** ((spl - CALIB_REF) / 20)

def create_dichotic_pitch(freq, bw=BANDWIDTH, flip='right'):
    wn = np.random.normal(size=N_SAMPLES)
    F  = fft(wn)
    f_axis = fftfreq(N_SAMPLES, 1/SAMPLE_RATE)
    band = ((abs(f_axis)>freq-bw) & (abs(f_axis)<freq+bw))
    F[band] *= -1
    flipped = np.real(ifft(F))
    m = max(np.max(abs(wn)), np.max(abs(flipped)))
    wn /= m; flipped /= m
    if flip=='left':
        stereo = np.column_stack((flipped, wn))
    elif flip=='right':
        stereo = np.column_stack((wn, flipped))
    else:
        stereo = np.column_stack((wn, wn))
    return stereo.astype(np.float32)

def save_stim_and_analysis(stereo, freq, flip, comp_flag, outdir):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_path = os.path.join(outdir, f"stim_{freq}Hz_{flip}_{ts}.wav")
    write(wav_path, SAMPLE_RATE, (stereo*32767).astype(np.int16))
    ch = 1 if flip=='right' else 0
    F_spec = fft(stereo[:,ch])
    mags = 20*np.log10(np.abs(F_spec)+1e-12)
    phs  = np.angle(F_spec)
    freqs= fftfreq(N_SAMPLES,1/SAMPLE_RATE)
    band = ((abs(freqs)>freq-BANDWIDTH) & (abs(freqs)<freq+BANDWIDTH))
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
    'Session #:':'001',
    'Hearing issues?':'n',
    'Neuro history?':'n',
    'Implants?':'n',
    'Medications?':'n',
    'Pregnant?':'n'
}
dlg = gui.DlgFromDict(scr, title="Pre-session Screening")
if not dlg.OK:
    core.quit()
pid, session = scr['Participant ID:'], scr['Session #:']
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
csv_path = os.path.join(out_dir,f"{pid}_{session}.csv")
header = (
    ['block','trial','phase','type','freq','heard','conf','rt',
     'ear','pitch','correct','flipped','ear_correct']
    + [f"L{f}" for f in sorted({20,250,400,600,800,1000,2000,4000,8000,16000,20000})]
    + [f"R{f}" for f in sorted({20,250,400,600,800,1000,2000,4000,8000,16000,20000})]
    + ['hear_issues','neuro','implants','meds','preg','eeg_imp','vol_ok','break_ok','compensated']
)
with open(csv_path,'w') as f:
    f.write(','.join(header)+'
')
data_f = open(csv_path,'a')

# 5) WINDOW SETUP
win = visual.Window(fullscr=False, color='black')
txt = visual.TextStim(win, color='white', height=0.08, wrapWidth=1.5)
fix = visual.TextStim(win, text='+', color='white', height=0.1)

# 6) EEG IMPEDANCE & VOLUME CALIBRATION
imp = {'Electrodes <5kΩ & stable?':'y'}
dlg2 = gui.DlgFromDict(imp, title='EEG Impedance')
if not dlg2.OK:
    core.quit()
eeg_ok = imp[next(iter(imp))]
volume = 0.25
while True:
    txt.text = (
        f"Volume: {int(volume*100)}%\n"
        "▲/▼ to adjust    T=play tone\n"
        "SPACE=OK    S=skip tests    ESC=quit"
    )
    txt.draw(); win.flip()
    keys = event.waitKeys(keyList=['up','down','t','space','s','escape'])
    if 'up' in keys:
        volume = min(1.0, volume+0.05)
    elif 'down' in keys:
        volume = max(0.0, volume-0.05)
    elif 't' in keys:
        t = np.linspace(0, DURATION, N_SAMPLES, False)
        tone = np.sin(2*np.pi*1000*t).astype(np.float32)
        snd = sound.Sound(value=np.column_stack((tone,tone)), sampleRate=SAMPLE_RATE, stereo=True)
        snd.setVolume(volume); snd.play(); core.wait(DURATION)
    elif 'escape' in keys:
        core.quit()
    elif 's' in keys:
        skip_hp = skip_hearing = True
        break
    elif 'space' in keys:
        skip_hp = skip_hearing = False
        break

# 7) HEADPHONE CHECK
if not skip_hp:
    t = np.linspace(0, DURATION, N_SAMPLES, endpoint=False)
    cont = sound.Sound(value=np.column_stack((np.sin(2*np.pi*1000*t),)*2),
                       sampleRate=SAMPLE_RATE, stereo=True)
    cont.setVolume(volume); cont.play(loops=-1)
    while True:
        txt.text = f"HP Volume: {int(volume*100)}%\n▲/▼ adjust    SPACE=begin   ESC=quit"
        txt.draw(); win.flip()
        keys = event.waitKeys(keyList=['up','down','space','escape'])
        if 'up' in keys:
            volume = min(1.0, volume+0.05); cont.setVolume(volume)
        elif 'down' in keys:
            volume = max(0.0, volume-0.05); cont.setVolume(volume)
        elif 'escape' in keys:
            cont.stop(); core.quit()
        elif 'space' in keys:
            break
    cont.stop()
    passed = True
    for _ in range(3):
        side = random.choice(['left','right'])
        stereo_tone = np.zeros((N_SAMPLES,2),dtype=np.float32)
        stereo_tone[:,0 if side=='left' else 1] = np.sin(2*np.pi*1000*t)
        s = sound.Sound(value=stereo_tone, sampleRate=SAMPLE_RATE, stereo=True)
        s.setVolume(volume); s.play(); core.wait(DURATION)

        txt.text = "Which ear? (L/R)    ESC=quit"
        txt.draw(); win.flip()
        ans = event.waitKeys(keyList=['left','right','escape'])[0]
        if ans=='escape': core.quit()
        if ans != side: passed = False

    if not passed:
        txt.text = "Headphone check FAILED – please retry."
        txt.draw(); win.flip(); core.wait(1.5)
else:
    txt.text = "Headphone test SKIPPED"
    txt.draw(); win.flip(); core.wait(1.0)

# 8) PURE-TONE HEARING TEST
txt.text = "PURE-TONE TEST: SPACE=Begin   S=Skip   ESC=Quit"
txt.draw(); win.flip()
k = event.waitKeys(keyList=['space','s','escape'])[0]
if k=='escape': core.quit()
skip_hearing = (k=='s')

pure_freqs = sorted({20,250,400,600,800,1000,2000,4000,8000,16000,20000})
thresholds = {'Left':{}, 'Right':{}}

if not skip_hearing:
    for ear in ['Left','Right']:
        ch = 0 if ear=='Left' else 1
        txt.text = f"{ear} ear: SPACE to begin"
        txt.draw(); win.flip()
        event.waitKeys(keyList=['space'])
        for f in pure_freqs:
            lvl = 50
            heard_once = False
            while True:
                amp = hl_to_amplitude(f, lvl)
                tvec = np.linspace(0, DURATION, N_SAMPLES, endpoint=False)
                tone = np.zeros((N_SAMPLES,2), dtype=np.float32)
                tone[:,ch] = amp * np.sin(2*np.pi*f*tvec)
                s = sound.Sound(value=tone, sampleRate=SAMPLE_RATE, stereo=True)
                s.setVolume(volume); s.play(); core.wait(DURATION)

                txt.text = f"Hear {f} Hz @ {lvl} dB? (Y/N)   ESC=quit"
                txt.draw(); win.flip()
                r = event.waitKeys(keyList=['y','n','escape'])[0]
                if r=='escape': core.quit()
                if r=='y':
                    heard_once = True
                    lvl = max(0, lvl - 10)
                else:
                    # first no moves up; second no (after any yes) moves on
                    if heard_once:
                        lvl += 5
                        break
                    lvl += 5
                lvl = int(np.clip(lvl, 0, 100))
            thresholds[ear][f] = lvl

# write zero-block screening row
row0 = ['0','0','Screen','NA','NA','NA','NA','NA','NA','NA','NA','NA','NA']
row0 += [thresholds['Left'].get(f,'')  for f in pure_freqs]
row0 += [thresholds['Right'].get(f,'') for f in pure_freqs]
row0 += [hearing_issues, neuro_history, implants, medications,
         pregnant, eeg_ok, 'yes', 'NA']
data_f.write(','.join(map(str,row0)) + '\n')
data_f.flush()

# 9) PRACTICE BLOCK
txt.text = "PRACTICE BLOCK: SPACE=continue"
txt.draw(); win.flip()
event.waitKeys(keyList=['space'])
for f in [400, 800]:
    stim = create_dichotic_pitch(f, bw=50, flip='right')
    snd = sound.Sound(value=stim, sampleRate=SAMPLE_RATE, stereo=True)
    snd.setVolume(volume); snd.play(); core.wait(DURATION)
    txt.text="Did you hear a pitch? (Y/N)"
    txt.draw(); win.flip()
    event.waitKeys(keyList=['y','n'])
    core.wait(0.5)

# 10) MAIN EXPERIMENT
def run_phase(compensated):
    phase = "Compensated" if compensated else "Uncompensated"
    txt.text = f"{phase} phase: SPACE=begin"
    txt.draw(); win.flip()
    event.waitKeys(keyList=['space'])

    for blk in range(1, NUM_BLOCKS+1):
        tests = random.sample([f for f in FREQUENCIES if f!=CONTROL_FREQ], 3)
        seq   = [CONTROL_FREQ] + tests + [CONTROL_FREQ]
        for tr, f in enumerate(seq, 1):
            typ = 'Control' if f==CONTROL_FREQ else 'Test'
            flip = random.choice(['left','right']) if typ=='Test' else 'none'
            stim = create_dichotic_pitch(f, flip=flip)
            if compensated and typ=='Test':
                lvl = thresholds[flip.capitalize()].get(f,50) + SL_TARGET
                stim *= hl_to_amplitude(f, lvl)
            if typ=='Test':
                save_stim_and_analysis(stim, f, flip, compensated, stim_dir)

            # fixation
            fix.draw(); win.flip(); core.wait(1.0)

            # play stimulus
            snd = sound.Sound(value=stim, sampleRate=SAMPLE_RATE, stereo=True)
            snd.setVolume(volume); snd.play(); core.wait(DURATION)

            # prompt immediately
            txt.text = "Did you hear a pitch? (Y/N)"
            txt.draw(); win.flip()
            key, rt = event.waitKeys(keyList=['y','n'], timeStamped=core.Clock())[0]

            # confidence
            txt.text = "Confidence (1–5)"
            txt.draw(); win.flip()
            conf = event.waitKeys(keyList=[str(i) for i in range(1,6)])[0]

            if key=='y':
                txt.text = "Which ear? (L/R/Both)"; txt.draw(); win.flip()
                ear = event.waitKeys(keyList=['left','right','b'])[0]
                txt.text = "High or Low? (H/L)"; txt.draw(); win.flip()
                pitch = event.waitKeys(keyList=['h','l'])[0]
            else:
                ear, pitch = 'NA','NA'

            corr = 'yes' if ((typ=='Test' and key=='y') or (typ=='Control' and key=='n')) else 'no'
            ear_corr = 'yes' if flip==ear else 'no' if flip in ['left','right'] else 'NA'
            row = [
                blk, tr, phase, typ, f, key, conf, f"{rt:.3f}",
                ear, pitch, corr, flip, ear_corr
            ]
            row += [''] * (len(pure_freqs)*2)
            row += [hearing_issues, neuro_history, implants,
                    medications, pregnant, eeg_ok, '', '', compensated]

            # break handling
            if blk < NUM_BLOCKS and tr == len(seq):
                txt.text = "Short break: SPACE=continue   N=end"
                txt.draw(); win.flip()
                b = event.waitKeys(keyList=['space','n'])[0]
                break_ok = 'yes' if b=='space' else 'no'
                br_row = [blk,0,phase,'Break'] + ['']*9
                br_row += [''] * (len(pure_freqs)*2)
                br_row += [hearing_issues, neuro_history, implants,
                           medications, pregnant, eeg_ok, '', break_ok, compensated]
                data_f.write(','.join(map(str,br_row))+'\n')
                data_f.flush()
                if b=='n':
                    txt.text = "Experiment ended early. Thank you!"
                    txt.draw(); win.flip(); core.wait(2.0)
                    core.quit()

            data_f.write(','.join(map(str,row)) + '\n')
            data_f.flush()
            core.wait(0.5)

# 11) RUN
run_phase(compensated=True)
run_phase(compensated=False)

# 12) CLEANUP
data_f.close()
txt.text = "Experiment complete. Thank you!"
txt.draw(); win.flip()
core.wait(2.0)
win.close()
core.quit()
