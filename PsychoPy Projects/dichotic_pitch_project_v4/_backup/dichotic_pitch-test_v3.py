#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dichotic-Pitch EEG Experiment
– Pre-session screening
– EEG impedance & volume calibration (with T-key test tone)
– Headphone check (continuous tone, adjustable, skip with S)
– Pure-tone hearing threshold test (skip with S)
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
    # re-create band mask
    bw = BANDWIDTH
    band = ((abs(freqs)>freq-bw) & (abs(freqs)<freq+bw))
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
if not getattr(dlg, 'OK', getattr(dlg, 'ok', False)):
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
header = [
 'block','trial','phase','type','freq','heard','conf','rt',
 'ear','pitch','correct','flipped','ear_correct',
 'L20','L250','L400','L600','L800','L1000','L2000','L4000','L8000','L16000','L20000',
 'R20','R250','R400','R600','R800','R1000','R2000','R4000','R8000','R16000','R20000',
 'hear_issues','neuro','implants','meds','preg','eeg_imp','vol_ok','break_ok','compensated'
]
data_f = open(csv_path,'w')
data_f.write(','.join(header)+'\n')

# 5) WINDOW SETUP
win = visual.Window(fullscr=False, color='black')
txt = visual.TextStim(win, color='white', height=0.08, wrapWidth=1.5)
fix = visual.TextStim(win, text='+', color='white', height=0.1)

# 6) EEG IMPEDANCE & VOLUME CALIBRATION
imp = {'Electrodes <5kΩ & stable?':'y'}
dlg2 = gui.DlgFromDict(imp, title='EEG Impedance')
if not getattr(dlg2,'OK',getattr(dlg2,'ok',False)): core.quit()
eeg_ok = imp[list(imp.keys())[0]]
volume = 0.25  # start at 25%

# show instructions + adjustable volume with T-test tone
while True:
    txt.text = (
        f"Volume: {int(volume*100)}%\n"
        "▲ Increase   ▼ Decrease\n"
        "T = Play test tone\n"
        "SPACE = OK   S = Skip tests   ESC = Quit"
    )
    txt.draw(); win.flip()
    keys = event.waitKeys(keyList=['up','down','t','space','s','escape'])
    if 'up' in keys:
        volume = min(1.0, volume + 0.05)
    elif 'down' in keys:
        volume = max(0.0, volume - 0.05)
    elif 't' in keys:
        # play 1 kHz tone for 1 s at current volume
        t = np.linspace(0, DURATION, N_SAMPLES, False)
        test_tone = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
        stereo = np.column_stack((test_tone, test_tone))
        snd = sound.Sound(value=stereo, sampleRate=SAMPLE_RATE, stereo=True)
        snd.setVolume(volume)
        snd.play()
        core.wait(DURATION)
    elif 'escape' in keys:
        core.quit()
    elif 's' in keys:
        skip_hp = True
        skip_hearing = True
        break
    elif 'space' in keys:
        skip_hp = False
        skip_hearing = False
        break

# 7) HEADPHONE CHECK (continuous tone, adjustable, skip with S)
if not skip_hp:
    t = np.linspace(0, DURATION, N_SAMPLES, endpoint=False)
    pure = np.sin(2 * np.pi * 1000 * t)
    pure_stereo = np.column_stack((pure, pure)).astype(np.float32)

    cont_sound = sound.Sound(value=pure_stereo,
                             sampleRate=SAMPLE_RATE,
                             stereo=True)
    cont_sound.setVolume(volume)
    cont_sound.play(loops=-1)

    while True:
        txt.text = (
            f"HEADPHONE TEST VOLUME: {int(volume*100)}%\n"
            "▲ Increase  ▼ Decrease\n"
            "SPACE = Begin trials\n"
            "ESC = Quit"
        )
        txt.draw(); win.flip()
        keys = event.waitKeys(keyList=['up','down','space','escape'])
        if 'up' in keys:
            volume = min(1.0, volume + 0.05)
            cont_sound.setVolume(volume)
        elif 'down' in keys:
            volume = max(0.0, volume - 0.05)
            cont_sound.setVolume(volume)
        elif 'escape' in keys:
            cont_sound.stop(); core.quit()
        elif 'space' in keys:
            break

    cont_sound.stop()

    passed = True
    for _ in range(3):
        side = random.choice(['left','right'])
        tone = np.zeros((N_SAMPLES,2), dtype=np.float32)
        ch = 0 if side=='left' else 1
        tone[:,ch] = np.sin(2*np.pi*1000*t)
        s = sound.Sound(value=tone, sampleRate=SAMPLE_RATE, stereo=True)
        s.setVolume(volume); s.play(); core.wait(DURATION)

        txt.text = "Which ear? (L/R/Both)  ESC=Quit"
        txt.draw(); win.flip()
        ans = event.waitKeys(keyList=['left','right','both','escape'])[0]
        if ans=='escape': core.quit()
        if ans not in [side,'both']:
            passed=False

    if not passed:
        txt.text = "Headphone check FAILED – retrying..."
        txt.draw(); win.flip(); core.wait(1.5)
else:
    txt.text = "Headphone test SKIPPED"
    txt.draw(); win.flip(); core.wait(1.0)

# 8) PURE-TONE HEARING TEST (skip with S)
txt.text = "PURE-TONE TEST\nSPACE=Begin  S=Skip  ESC=Quit"
txt.draw(); win.flip()
key = event.waitKeys(keyList=['space','s','escape'])[0]
if key=='escape': core.quit()
if key=='s':
    skip_hearing = True

pure_freqs = sorted([20, 250, 400, 600, 800, 1000, 2000, 4000, 8000, 16000, 20000])
thresholds = {'Left':{}, 'Right':{}}

if not skip_hearing:
    for ear in ['Left','Right']:
        ch = 0 if ear=='Left' else 1
        txt.text = f"Pure-tone test: {ear} ear. SPACE to begin"
        txt.draw(); win.flip(); event.waitKeys(['space'])
        for f in pure_freqs:
            lvl = 50; heard_once=False
            while True:
                amp = hl_to_amplitude(f, lvl)
                t = np.linspace(0, DURATION, N_SAMPLES)
                tone = np.zeros((N_SAMPLES,2), dtype=np.float32)
                tone[:,ch] = amp * np.sin(2*np.pi*f*t)
                s = sound.Sound(value=tone, sampleRate=SAMPLE_RATE, stereo=True)
                s.setVolume(volume); s.play(); core.wait(DURATION)

                txt.text = f"Hear {f}Hz at {lvl} dB? (y/n) ESC=Quit"
                txt.draw(); win.flip()
                k = event.waitKeys(keyList=['y','n','escape'])[0]
                if k=='escape': core.quit()
                if k=='y':
                    heard_once=True
                    if lvl<=0:
                        # pink noise false-positive check
                        pink = np.random.randn(N_SAMPLES).astype(np.float32)
                        s2 = sound.Sound(value=np.column_stack((pink,pink)),
                                         sampleRate=SAMPLE_RATE, stereo=True)
                        s2.setVolume(volume*1.2)  # +3dB boost
                        s2.play(); core.wait(DURATION)
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
row0 += [ thresholds['Left'].get(f,'') for f in pure_freqs ]
row0 += [ thresholds['Right'].get(f,'') for f in pure_freqs ]
row0 += [ hearing_issues, neuro_history, implants, medications, pregnant, eeg_ok, 'yes', 'NA' ]
data_f.write(','.join(map(str,row0))+'\n')
data_f.flush()

# 9) PRACTICE BLOCK
txt.text = "PRACTICE BLOCK\nSPACE to continue"
txt.draw(); win.flip(); event.waitKeys(['space'])
for f in [400,800]:
    stim = create_dichotic_pitch(f, bw=50, flip='right')
    sound.Sound(value=stim, sampleRate=SAMPLE_RATE, stereo=True).setVolume(volume).play()
    core.wait(DURATION)
    txt.text = "Did you hear a pitch? (y/n)";
    txt.draw(); win.flip(); event.waitKeys(['y','n'])
    core.wait(0.5)

# 10) MAIN EXPERIMENT
def run_phase(compensated):
    phase = "Compensated" if compensated else "Uncompensated"
    txt.text = f"Starting {phase} phase\nSPACE to begin"
    txt.draw(); win.flip(); event.waitKeys(['space'])
    for blk in range(1,NUM_BLOCKS+1):
        tests = random.sample([f for f in FREQUENCIES if f!=CONTROL_FREQ],3)
        seq   = [CONTROL_FREQ] + tests + [CONTROL_FREQ]
        for tr,f in enumerate(seq,1):
            typ = 'Control' if f==CONTROL_FREQ else 'Test'
            flip = random.choice(['left','right']) if typ=='Test' else 'none'
            stim = create_dichotic_pitch(f, flip=flip)
            if compensated and typ=='Test':
                lvl = thresholds[flip.capitalize()].get(f,50) + SL_TARGET
                amp = hl_to_amplitude(f, lvl)
                stim *= amp
            if typ=='Test':
                save_stim_and_analysis(stim, f, flip, compensated, stim_dir)
            fix.draw(); win.flip(); core.wait(1.0)
            sound.Sound(value=stim, sampleRate=SAMPLE_RATE, stereo=True).setVolume(volume).play()
            core.wait(DURATION)

            txt.text = "Hear a pitch? (y/n)"
            txt.draw(); win.flip()
            key, rt = event.waitKeys(keyList=['y','n'], timeStamped=core.Clock())[0]
            txt.text = "Confidence? (1–5)"
            txt.draw(); win.flip()
            conf = event.waitKeys(keyList=[str(i) for i in range(1,6)])[0]

            if key=='y':
                txt.text="Which ear? (L/R/Both)"; txt.draw(); win.flip()
                ear = event.waitKeys(keyList=['left','right','both'])[0]
                txt.text="High or Low? (H/L)"; txt.draw(); win.flip()
                pt  = event.waitKeys(keyList=['h','l'])[0]
            else:
                ear,pt = 'NA','NA'

            corr = 'yes' if (typ=='Test' and key=='y') or (typ=='Control' and key=='n') else 'no'
            ear_corr = 'yes' if flip==ear else 'no' if flip in ['left','right'] else 'NA'
            row = [
              blk, tr, phase, typ, f, key, conf, f"{rt:.3f}",
              ear, 'high' if pt=='h' else 'low' if pt=='l' else 'NA',
              corr, flip, ear_corr
            ]
            row += ['']*22
            row += [hearing_issues, neuro_history, implants,
                    medications, pregnant, eeg_ok, '', compensated]
            data_f.write(','.join(map(str,row))+'\n')
            data_f.flush()
            core.wait(0.5)

        if blk<NUM_BLOCKS:
            txt.text="Short break – SPACE to continue or N to end"
            txt.draw(); win.flip()
            k=event.waitKeys(keyList=['space','n'])[0]
            if k=='n':
                data_f.write(f"{blk},0,{phase},Break,,,,,,,,,,,,,,,,no,{compensated}\n")
                data_f.flush()
                core.quit()
            else:
                data_f.write(f"{blk},0,{phase},Break,,,,,,,,,,,,,,,,yes,{compensated}\n")
                data_f.flush()

# 11) RUN
run_phase(compensated=True)
run_phase(compensated=False)

# 12) CLEANUP
data_f.close()
txt.text="Experiment complete. Thank you!"
txt.draw(); win.flip()
core.wait(2.0)
win.close()
core.quit()
