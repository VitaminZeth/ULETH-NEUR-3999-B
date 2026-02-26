#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dichotic-Pitch EEG Experiment (vX.X)
  • 1) fixed undefined `band` in FFT CSV
  • 2) persistent core.Clock() for RTs
  • 3) bullet-proof dlg.OK checks
  • 4) sample-rate fallback wrapper
  • 5) every CSV row now exactly matches header
  • 6) explicit vol_ok & break_ok columns
  • 7) one Sound-factory with backend fallback
"""

from psychopy import prefs
prefs.hardware['audioLib'] = ['sounddevice']  # force sounddevice

from psychopy import sound, core, event, gui, visual
import numpy as np
from scipy.fft import fft, ifft, fftfreq
from scipy.io.wavfile import write
import pandas as pd
import random, os
from datetime import datetime

# --- CONSTANTS ---
SAMPLE_RATE = 44100
DURATION    = 1.0
N_SAMPLES   = int(SAMPLE_RATE * DURATION)
BANDWIDTH   = 20
FREQUENCIES = [250, 400, 600, 800, 1000]
CONTROL_FREQ= 500
NUM_BLOCKS  = 2
RETSPL      = {250:26.5,500:13.5,1000:7.5,2000:9.0,4000:9.0,8000:13.5}
CALIB_REF   = 94.0   # dB SPL full-scale
SL_TARGET   = 40     # above threshold

# persistent clock for all RTs
rt_clock = core.Clock()

# --- SOUND FACTORY w/ fallback ---
def make_sound(stereo_array):
    try:
        return sound.Sound(value=stereo_array,
                           sampleRate=SAMPLE_RATE,
                           stereo=True)
    except Exception:
        return sound.Sound(value=stereo_array,
                           stereo=True)

# --- Utilities ---
def hl_to_amplitude(freq, hl_dbhl):
    spl = hl_dbhl + RETSPL[freq]
    return 10 ** ((spl - CALIB_REF) / 20)

def create_dichotic_pitch(freq, bw=BANDWIDTH, flip='right'):
    wn = np.random.normal(size=N_SAMPLES)
    F  = fft(wn)
    f_axis = fftfreq(N_SAMPLES, 1/SAMPLE_RATE)
    band_mask = (abs(f_axis)>freq-bw) & (abs(f_axis)<freq+bw)
    F[band_mask] *= -1
    flipped = np.real(ifft(F))
    m = max(np.max(abs(wn)), np.max(abs(flipped)))
    wn /= m; flipped /= m
    if flip=='left':
        stereo = np.column_stack((flipped, wn))
    elif flip=='right':
        stereo = np.column_stack((wn, flipped))
    else:
        stereo = np.column_stack((wn, wn))
    return stereo.astype(np.float32), band_mask

# buffer for FFT analysis logs
fft_logs = []

def save_stim_and_analysis(stereo, freq, flip, comp_flag, outdir, band_mask):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_path = os.path.join(outdir, f"stim_{freq}Hz_{flip}_{ts}.wav")
    write(wav_path, SAMPLE_RATE, (stereo*32767).astype(np.int16))
    # FFT on flipped channel
    ch = 1 if flip=='right' else 0
    F = fft(stereo[:,ch])
    freqs = fftfreq(N_SAMPLES,1/SAMPLE_RATE)
    mags = 20*np.log10(np.abs(F)+1e-12)
    phs  = np.angle(F)
    df = pd.DataFrame({
        'Frequency':     freqs[:N_SAMPLES//2],
        'Magnitude_dB':  mags[:N_SAMPLES//2],
        'Phase_rad':     phs[:N_SAMPLES//2],
        'FlippedBand':   band_mask[:N_SAMPLES//2],
        'Channel':       flip,
        'Compensated':   comp_flag
    })
    csv_path = wav_path.replace('.wav','.csv')
    df.to_csv(csv_path, index=False)
    # optionally buffer and write later:
    fft_logs.append((csv_path, df))

# --- 1) Pre-session screening ---
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
# robust OK check
ok = getattr(dlg, 'OK', getattr(dlg,'ok', None))
if ok is False or ok is None:
    core.quit()
pid, session = scr['Participant ID:'], scr['Session #:']
hearing_issues = scr['Hearing issues?']
neuro_history  = scr['Neuro history?']
implants       = scr['Implants?']
medications    = scr['Medications?']
pregnant       = scr['Pregnant?']

# --- 2) Directories & CSV header ---
out_dir = f"data/{pid}_{session}"
stim_dir= os.path.join(out_dir,'stimuli')
os.makedirs(stim_dir, exist_ok=True)
header = [
 'block','trial','phase','type','freq','heard','conf','rt',
 'ear','pitch','correct','flipped','ear_correct',
 'L250','L500','L1000','L2000','L4000','L8000',
 'R250','R500','R1000','R2000','R4000','R8000',
 'hear_issues','neuro','implants','meds','preg',
 'eeg_imp','vol_ok','break_ok','compensated'
]
csv_path = os.path.join(out_dir,f"{pid}_{session}.csv")
data_f = open(csv_path,'w')
data_f.write(','.join(header)+'\n')

# --- 3) Window & stimuli text ---
win = visual.Window(fullscr=False, color='black')
txt = visual.TextStim(win, color='white', height=0.08, wrapWidth=1.5)
fix = visual.TextStim(win, text='+', color='white', height=0.1)

# --- 4) EEG impedance & volume ---
imp = {'Electrodes <5kΩ & stable?':'y'}
dlg2 = gui.DlgFromDict(imp, title='EEG Impedance')
ok2 = getattr(dlg2,'OK',getattr(dlg2,'ok', None))
if ok2 is False or ok2 is None:
    core.quit()
eeg_ok = imp[list(imp.keys())[0]]

txt.text = "Adjust volume comfortably, then press SPACE"
txt.draw(); win.flip()
event.waitKeys(keyList=['space'])
vol_ok = 'yes'  # we know they did it

# --- 5) Headphone check (loop until pass) ---
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
        s = make_sound(tone)
        s.play(); core.wait(DURATION)
        txt.text = "Which ear? (LEFT/RIGHT) or ESC"
        txt.draw(); win.flip()
        resp = event.waitKeys(keyList=['left','right','escape'])[0]
        if resp=='escape': core.quit()
        if resp!=side:
            passed=False
    if passed: break
    txt.text="Headphone check FAILED. Retrying..."
    txt.draw(); win.flip(); core.wait(1.5)

# --- 6) Pure-tone hearing test ---
txt.text="PURE-TONE TEST\nPress SPACE to begin or ESC to quit"
txt.draw(); win.flip()
k = event.waitKeys(keyList=['space','escape'])[0]
if k=='escape': core.quit()

pure_freqs = [250,500,1000,2000,4000,8000]
thresholds = {'Left':{},'Right':{}}
for ear in ['Left','Right']:
    ch = 0 if ear=='Left' else 1
    txt.text = f"Pure-tone test: {ear} ear. Press SPACE"
    txt.draw(); win.flip()
    event.waitKeys(['space'])
    for f in pure_freqs:
        lvl = 50; heard_once=False
        while True:
            amp = hl_to_amplitude(f, lvl)
            t = np.linspace(0,DURATION,N_SAMPLES)
            tone = np.zeros((N_SAMPLES,2))
            tone[:,ch] = amp*np.sin(2*np.pi*f*t)
            make_sound(tone).play()
            core.wait(DURATION)
            rt_clock.reset()
            txt.text = f"Hear {f}Hz at {lvl} dB? (y/n)"
            txt.draw(); win.flip()
            resp, rt = event.waitKeys(keyList=['y','n'], timeStamped=rt_clock)[0]
            if resp=='y':
                heard_once=True
                lvl -= 10
            else:
                if heard_once:
                    lvl += 5
                    break
                lvl += 5
            lvl = int(np.clip(lvl,0,100))
        thresholds[ear][f] = lvl

# write screening row
row0 = ['0','0','Screen','NA','NA','NA','NA','NA','NA','NA','NA','NA','NA']
row0 += [thresholds['Left'][f]  for f in pure_freqs]
row0 += [thresholds['Right'][f] for f in pure_freqs]
row0 += [hearing_issues, neuro_history, implants, medications, pregnant, eeg_ok, vol_ok, 'NA', 'NA']
data_f.write(','.join(map(str,row0))+'\n')
data_f.flush()

# --- 7) Practice block ---
txt.text="PRACTICE BLOCK\nPress SPACE to continue"
txt.draw(); win.flip()
event.waitKeys(['space'])
for f in [400,800]:
    stim, _ = create_dichotic_pitch(f, bw=50, flip='right')
    make_sound(stim).play()
    core.wait(DURATION)
    txt.text="Did you hear a pitch? (y/n)"
    txt.draw(); win.flip()
    event.waitKeys(['y','n'])
    core.wait(0.5)

# --- 8) Main experiment phases ---
def run_phase(compensated):
    phase = "Compensated" if compensated else "Uncompensated"
    txt.text = f"Starting {phase} phase\nPress SPACE"
    txt.draw(); win.flip()
    event.waitKeys(['space'])
    for blk in range(1, NUM_BLOCKS+1):
        tests = random.sample([f for f in FREQUENCIES if f!=CONTROL_FREQ], 3)
        seq = [CONTROL_FREQ] + tests + [CONTROL_FREQ]
        for tr, f in enumerate(seq,1):
            typ  = 'Control' if f==CONTROL_FREQ else 'Test'
            flip = random.choice(['left','right']) if typ=='Test' else 'none'
            stim, band_mask = create_dichotic_pitch(f, flip=flip)
            if compensated and typ=='Test':
                lvl = thresholds[flip.capitalize()][f] + SL_TARGET
                stim *= hl_to_amplitude(f, lvl)
            if typ=='Test':
                save_stim_and_analysis(stim, f, flip, compensated, stim_dir, band_mask)

            # trial
            fix.draw(); win.flip(); core.wait(1.0)
            make_sound(stim).play()
            core.wait(DURATION)

            rt_clock.reset()
            txt.text="Hear a pitch? (y/n)"
            txt.draw(); win.flip()
            heard, rt = event.waitKeys(keyList=['y','n'], timeStamped=rt_clock)[0]

            txt.text="Confidence? (1–5)"
            txt.draw(); win.flip()
            conf = event.waitKeys(keyList=[str(i) for i in range(1,6)])[0]

            if heard=='y':
                txt.text="Which ear? (left/right)"; txt.draw(); win.flip()
                ear = event.waitKeys(keyList=['left','right'])[0]
                txt.text="High or Low? (h/l)"; txt.draw(); win.flip()
                pt  = event.waitKeys(keyList=['h','l'])[0]
            else:
                ear, pt = 'NA','NA'

            corr     = 'yes' if (typ=='Test' and heard=='y') or (typ=='Control' and heard=='n') else 'no'
            ear_corr = 'yes' if flip==ear else ('no' if flip in ['left','right'] else 'NA')

            # build exactly len(header) row
            row = [
              blk, tr, phase, typ, f, heard, conf, f"{rt:.3f}",
              ear,
              'high' if pt=='h' else 'low' if pt=='l' else 'NA',
              corr, flip, ear_corr
            ]
            row += [''] * 12          # placeholder for thresholds
            row += [
              hearing_issues, neuro_history, implants,
              medications, pregnant, eeg_ok, vol_ok,
              'NA', compensated
            ]
            assert len(row)==len(header), f"Row len {len(row)} != header {len(header)}"
            data_f.write(','.join(map(str,row))+'\n')
            data_f.flush()
            core.wait(0.5)

        # break
        if blk<NUM_BLOCKS:
            txt.text="Short break – SPACE to continue or N to end"
            txt.draw(); win.flip()
            b = event.waitKeys(keyList=['space','n'])[0]
            break_ok = 'yes' if b=='space' else 'no'
            data_f.write(','.join(map(str([
                blk, 0, phase, 'Break','NA','NA','NA','NA',
                'NA','NA','NA','NA','NA'] +
                ['']*12 +
                [hearing_issues, neuro_history, implants,
                 medications, pregnant, eeg_ok, vol_ok,
                 break_ok, compensated]
            ))) + '\n')
            data_f.flush()

# run both phases
run_phase(compensated=True)
run_phase(compensated=False)

# --- 9) Cleanup & write FFT logs (optional) ---
for path, df in fft_logs:
    # if you want to re-write or merge logs in bulk, do it here
    pass

data_f.close()
txt.text="Experiment complete. Thank you!"
txt.draw(); win.flip()
core.wait(2.0)
win.close()
core.quit()
