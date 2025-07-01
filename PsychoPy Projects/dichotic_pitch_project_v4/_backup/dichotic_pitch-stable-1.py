#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dichotic-Pitch EEG Experiment
Fully merged, crash-fixed version.
"""

from psychopy import prefs
prefs.hardware['audioLib'] = ['sounddevice']

from psychopy import sound, core, event, gui, visual
import numpy as np
from scipy.fft import fft, ifft, fftfreq
from scipy.io.wavfile import write
import pandas as pd
import random, os
from datetime import datetime

# --- Constants ---
SAMPLE_RATE = 44100
DURATION    = 1.0
N_SAMPLES   = int(SAMPLE_RATE * DURATION)
BANDWIDTH   = 20

# Only frequencies we actually support in RETSPL
RETSPL      = {20:20.0,  25:20.0,  50:20.0,
               250:26.5, 500:13.5,1000:7.5,
               2000:9.0, 4000:9.0, 8000:13.5,
               16000:15.0,20000:20.0}

CONTROL_FREQ= 500
FREQUENCIES = [250, 500, 1000, 2000, 4000, 8000]
EXTRA_FREQS = [20, 16000, 20000]  # hearing-test only

NUM_BLOCKS  = 2
CALIB_REF   = 94.0
SL_TARGET   = 40

# --- Helper functions ---
def hl_to_amplitude(freq, hl_dbhl):
    """Convert dB HL → normalized amplitude."""
    spl = hl_dbhl + RETSPL.get(freq, CALIB_REF)
    return 10 ** ((spl - CALIB_REF) / 20)

def create_dichotic_pitch(freq, bw=BANDWIDTH, flip='right'):
    wn = np.random.normal(size=N_SAMPLES)
    F  = fft(wn)
    f_axis = fftfreq(N_SAMPLES, 1/SAMPLE_RATE)
    band = ((np.abs(f_axis)>freq-bw)&(np.abs(f_axis)<freq+bw))
    F[band] *= -1
    flipped = np.real(ifft(F))
    m = max(np.max(np.abs(wn)), np.max(np.abs(flipped))) or 1
    wn   /= m; flipped /= m
    if flip=='left':
        stereo = np.column_stack((flipped, wn))
    elif flip=='right':
        stereo = np.column_stack((wn, flipped))
    else:
        stereo = np.column_stack((wn, wn))
    return stereo.astype(np.float32)

def save_stim_and_analysis(stereo, freq, flip, comp, outdir):
    """Save WAV + FFT CSV, including recomputing 'band' mask."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_path = os.path.join(outdir, f"stim_{freq}Hz_{flip}_{ts}.wav")
    write(wav_path, SAMPLE_RATE, (stereo*32767).astype(np.int16))

    # pick channel to FFT
    ch = 1 if flip=='right' else 0
    F     = fft(stereo[:,ch])
    mags  = 20*np.log10(np.abs(F)+1e-12)
    phs   = np.angle(F)
    f_axis= fftfreq(N_SAMPLES,1/SAMPLE_RATE)
    band  = ((np.abs(f_axis)>freq-BANDWIDTH)&(np.abs(f_axis)<freq+BANDWIDTH))

    df = pd.DataFrame({
      'Frequency':    f_axis[:N_SAMPLES//2],
      'Magnitude_dB': mags[:N_SAMPLES//2],
      'Phase_rad':    phs[:N_SAMPLES//2],
      'FlippedBand':  band[:N_SAMPLES//2],
      'Channel':      flip,
      'Compensated':  comp
    })
    df.to_csv(wav_path.replace('.wav','.csv'), index=False)


# === 1) PARTICIPANT SCREENING ===
scr = {
  'Participant ID:':'', 'Session #:':'001',
  'Hearing issues?':'n','Neuro history?':'n',
  'Implants?':'n','Medications?':'n','Pregnant?':'n'
}
dlg = gui.DlgFromDict(scr, title="Pre-session Screening")
if not dlg.OK: core.quit()
pid, session = scr['Participant ID:'], scr['Session #:']

# flatten
hearing_issues = scr['Hearing issues?']
neuro_history  = scr['Neuro history?']
implants       = scr['Implants?']
medications    = scr['Medications?']
pregnant       = scr['Pregnant?']


# === 2) FILE & LOG SETUP ===
out_dir = f"data/{pid}_{session}"
os.makedirs(out_dir, exist_ok=True)
stim_dir= os.path.join(out_dir,'stimuli')
os.makedirs(stim_dir, exist_ok=True)

# main CSV
csv_path = os.path.join(out_dir,f"{pid}_{session}.csv")
header = [
 'block','trial','phase','type','freq','heard','conf','rt',
 'ear','pitch','correct','flipped','ear_correct'
] + [f"L{f}" for f in RETSPL] + [f"R{f}" for f in RETSPL] + [
 'hear_issues','neuro','implants','meds','preg','eeg_ok','vol_ok','break_ok','compensated'
]
with open(csv_path,'w') as f:
    f.write(','.join(header)+'\n')

# false-positive log
fp_log = open(os.path.join(out_dir,'fp_log.csv'),'w')
fp_log.write('ear,freq,heard_pink\n')


# === 3) WINDOW & TEXT ===
win = visual.Window(fullscr=False, color='black')
txt = visual.TextStim(win, color='white', height=0.08, wrapWidth=1.5)
fix = visual.TextStim(win, text='+', color='white', height=0.1)


# === 4) EEG IMPEDANCE ===
imp = {'Electrodes <5kΩ & stable?':'y'}
dlg2 = gui.DlgFromDict(imp, title='EEG Impedance')
if not dlg2.OK: core.quit()
eeg_ok = imp[list(imp.keys())[0]]


# === 5) SOFTWARE-VOLUME SLIDER + SKIP + PINK NOISE ===
skip_headphone = False
volume = 0.25

# generate pink noise once
pn = np.random.normal(size=N_SAMPLES)
pn /= np.max(np.abs(pn)) or 1
pink_stereo = np.column_stack((pn,pn)).astype(np.float32)
pink_sound  = sound.Sound(value=pink_stereo, sampleRate=SAMPLE_RATE, stereo=True)
pink_sound.setVolume(volume)
pink_sound.play(loops=-1)

while True:
    txt.text = (
      f"Volume: {int(volume*100)}%   "
      "(Up/Down adjust, SPACE=OK, S=skip headphone)"
    )
    txt.draw(); win.flip()
    keys = event.waitKeys(keyList=['up','down','space','s','escape'])
    if 'up' in keys:
        volume = min(1.0, volume+0.05)
        pink_sound.setVolume(volume)
    elif 'down' in keys:
        volume = max(0.0, volume-0.05)
        pink_sound.setVolume(volume)
    elif 's' in keys:
        skip_headphone = True
        break
    elif 'space' in keys:
        break
    elif 'escape' in keys:
        core.quit()

pink_sound.stop()


# === 6) HEADPHONE CHECK ===
if not skip_headphone:
    while True:
        txt.text = "HEADPHONE CHECK\nSPACE=begin  ESC=quit"
        txt.draw(); win.flip()
        k = event.waitKeys(keyList=['space','escape'])[0]
        if k=='escape': core.quit()

        passed = True
        for _ in range(3):
            side = random.choice(['left','right'])
            tone = np.zeros((N_SAMPLES,2))
            t    = np.linspace(0,DURATION,N_SAMPLES)
            tone[:,0 if side=='left' else 1] = np.sin(2*np.pi*1000*t)
            s = sound.Sound(value=tone, sampleRate=SAMPLE_RATE, stereo=True)
            s.setVolume(volume); s.play()
            core.wait(DURATION)

            txt.text = "Which ear? (LEFT/RIGHT)  ESC=quit"
            txt.draw(); win.flip()
            ans = event.waitKeys(keyList=['left','right','escape'])[0]
            if ans=='escape': core.quit()
            if ans!=side: passed=False

        if passed:
            break
        else:
            txt.text = "Fail—retrying..."
            txt.draw(); win.flip(); core.wait(1.5)
else:
    txt.text = "Headphone SKIPPED"
    txt.draw(); win.flip(); core.wait(1.0)


# === 7) PURE-TONE HEARING TEST ===
txt.text = "PURE-TONE TEST\nSPACE=begin  ESC=quit"
txt.draw(); win.flip()
k = event.waitKeys(keyList=['space','escape'])[0]
if k=='escape': core.quit()

pure_freqs = list(RETSPL.keys())
thresholds   = {'Left':{}, 'Right':{}}

for ear in ['Left','Right']:
    ch = 0 if ear=='Left' else 1
    txt.text = f"{ear} ear – SPACE to begin"
    txt.draw(); win.flip()
    event.waitKeys(keyList=['space'])

    for f in pure_freqs + EXTRA_FREQS:
        # skip EXTRA_FREQS in threshold storage, but still test
        lvl = 50
        heard_once = False

        while True:
            amp = hl_to_amplitude(f, lvl)
            t   = np.linspace(0,DURATION,N_SAMPLES)
            tone= np.zeros((N_SAMPLES,2))
            tone[:,ch] = amp * np.sin(2*np.pi*f*t)
            s = sound.Sound(value=tone, sampleRate=SAMPLE_RATE, stereo=True)
            s.setVolume(volume); s.play()
            core.wait(DURATION)

            txt.text = f"Hear {f} Hz at {lvl} dB? (Y/N)"
            txt.draw(); win.flip()
            k = event.waitKeys(keyList=['y','n'])[0]

            if k=='y' and lvl==0:
                # pink-noise fallback @ +5 dB
                amp5 = hl_to_amplitude(f,5)
                pn2  = np.random.normal(size=N_SAMPLES)
                pn2 /= np.max(np.abs(pn2)) or 1
                tn2  = np.zeros((N_SAMPLES,2))
                tn2[:,ch] = amp5 * pn2
                s2 = sound.Sound(value=tn2, sampleRate=SAMPLE_RATE, stereo=True)
                s2.setVolume(volume); s2.play()
                core.wait(DURATION)

                txt.text = "Pink noise heard? (Y/N)"
                txt.draw(); win.flip()
                k2 = event.waitKeys(keyList=['y','n'])[0]

                # log false positive
                fp_log.write(f"{ear},{f},{k2}\n"); fp_log.flush()

            if k=='y':
                heard_once = True
                lvl -= 10
            else:
                if heard_once:
                    lvl += 5
                    break
                lvl += 5

            lvl = int(np.clip(lvl,0,100))

        if f in RETSPL:
            thresholds[ear][f] = lvl


# write initial thresholds row
with open(csv_path,'a') as data_f:
    row0 = ['0','0','Screen','NA','NA','NA','NA','NA','NA','NA','NA','NA','NA']
    row0 += [thresholds['Left'].get(f,'') for f in RETSPL]
    row0 += [thresholds['Right'].get(f,'') for f in RETSPL]
    row0 += [hearing_issues, neuro_history, implants,
             medications, pregnant, eeg_ok, 'yes', 'NA']
    data_f.write(','.join(map(str,row0))+'\n')


# === 8) PRACTICE ===
txt.text = "PRACTICE BLOCK\nSPACE to continue"
txt.draw(); win.flip()
event.waitKeys(keyList=['space'])
for f in [400, 800]:
    stim = create_dichotic_pitch(f, bw=50, flip='right')
    s = sound.Sound(value=stim, sampleRate=SAMPLE_RATE, stereo=True)
    s.setVolume(volume); s.play()
    core.wait(DURATION)
    txt.text="Pitch? (Y/N)"; txt.draw(); win.flip()
    event.waitKeys(keyList=['y','n']); core.wait(0.5)


# === 9) MAIN EXPERIMENT ===
def run_phase(compensated):
    phase = "Comp" if compensated else "Uncomp"
    txt.text = f"{phase} phase\nSPACE to begin"
    txt.draw(); win.flip()
    event.waitKeys(keyList=['space'])

    for blk in range(1,NUM_BLOCKS+1):
        seq = [CONTROL_FREQ] + random.sample(FREQUENCIES,3) + [CONTROL_FREQ]
        for tr, f in enumerate(seq,1):
            typ  = 'Control' if f==CONTROL_FREQ else 'Test'
            flip = random.choice(['left','right']) if typ=='Test' else 'none'
            stim = create_dichotic_pitch(f, flip=flip)

            if compensated and typ=='Test':
                lvl = thresholds[flip.capitalize()].get(f,0) + SL_TARGET
                stim *= hl_to_amplitude(f,lvl)

            if typ=='Test':
                save_stim_and_analysis(stim,f,flip,compensated,stim_dir)

            fix.draw(); win.flip(); core.wait(1.0)
            s = sound.Sound(value=stim, sampleRate=SAMPLE_RATE, stereo=True)
            s.setVolume(volume); s.play(); core.wait(DURATION)

            key, rt = event.waitKeys(keyList=['y','n'], timeStamped=core.Clock())[0]
            txt.text="Confidence? (1-5)"; txt.draw(); win.flip()
            conf = event.waitKeys(keyList=[str(i) for i in range(1,6)])[0]

            if key=='y':
                txt.text="Which ear?"; txt.draw(); win.flip()
                ear = event.waitKeys(keyList=['left','right'])[0]
                txt.text="High/Low?"; txt.draw(); win.flip()
                pt  = event.waitKeys(keyList=['h','l'])[0]
            else:
                ear,pt='NA','NA'

            corr     = ('yes' if (typ=='Test'and key=='y')or(typ=='Control'and key=='n') else 'no')
            ear_corr = ('yes' if flip==ear else 'no' if flip!='none' else 'NA')

            row = [
              blk,tr,phase,typ,f,key,conf,f"{rt:.3f}",ear,
              ('high' if pt=='h' else 'low' if pt=='l' else 'NA'),
              corr,flip,ear_corr
            ]
            row += ['']* (len(RETSPL)*2)
            row += [hearing_issues, neuro_history, implants,
                    medications, pregnant, eeg_ok,'',compensated]

            with open(csv_path,'a') as data_f:
                data_f.write(','.join(map(str,row))+'\n')
            core.wait(0.5)

        if blk<NUM_BLOCKS:
            txt.text="Break—SPACE to continue or N to end"
            txt.draw(); win.flip()
            b = event.waitKeys(keyList=['space','n'])[0]
            with open(csv_path,'a') as data_f:
                data_f.write(f"{blk},0,{phase},Break,,,,,,,,,,,,,,,{b=='space'},{compensated}\n")


run_phase(compensated=True)
run_phase(compensated=False)

# === CLEANUP ===
fp_log.close()
txt.text="All done—thank you!"
txt.draw(); win.flip()
core.wait(2.0)
win.close(); core.quit()
