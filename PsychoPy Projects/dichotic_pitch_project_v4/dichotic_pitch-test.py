from psychopy import prefs, sound, core, event, gui, visual
import numpy as np
import pandas as pd
import random, os
from scipy.fft import fft, ifft, fftfreq
from scipy.io.wavfile import write
from datetime import datetime

# =============================================================================
# 1) BACKEND & PREFS
# —————————————————————————————————————————————————————————————
prefs.hardware['audioLib'] = ['sounddevice']

# =============================================================================
# 2) DIALOG & FILE SETUP
# —————————————————————————————————————————————————————————————
exp_info = {'Participant': '', 'Session': '001'}
dlg = gui.DlgFromDict(exp_info, title="Dichotic Pitch EEG")
if not dlg.OK:
    core.quit()

os.makedirs('data',    exist_ok=True)
os.makedirs('stimuli', exist_ok=True)
fname = f"data/{exp_info['Participant']}_{exp_info['Session']}.csv"
data_file = open(fname, 'w')

header = [
    'block','trial','type','frequency',
    'heard','confidence','rt',
    'ear_perceived','pitch_type','correct',
    'channel_flipped','ear_correct'
]
data_file.write(','.join(header) + '\n')

# =============================================================================
# 3) WINDOW & TEXT
# —————————————————————————————————————————————————————————————
win   = visual.Window(fullscr=False, color='black')
instr = visual.TextStim(win, color='white', height=0.07, wrapWidth=1.5)
fix   = visual.TextStim(win, text='+', color='white', height=0.1)

# =============================================================================
# 4) AUDIO & EXPERIMENT PARAMETERS
# —————————————————————————————————————————————————————————————
SAMPLE_RATE  = 44100
DURATION     = 1.0
N_SAMPLES    = int(SAMPLE_RATE * DURATION)
BANDWIDTH    = 20

CONTROL_FREQ = 500
TEST_FREQS   = [250,400,600,800,1000]
NUM_BLOCKS   = 2

# =============================================================================
# 5) UTILITY FUNCTIONS
# —————————————————————————————————————————————————————————————
def create_dichotic_pitch(freq, flip_channel='right'):
    wn = np.random.normal(0,1,N_SAMPLES)
    F  = fft(wn)
    f  = fftfreq(N_SAMPLES, d=1/SAMPLE_RATE)
    for i,fi in enumerate(f):
        if abs(fi)>freq-BANDWIDTH and abs(fi)<freq+BANDWIDTH:
            F[i] *= -1
    flipped = np.real(ifft(F))
    m = max(np.max(np.abs(wn)), np.max(np.abs(flipped)))
    wn_n = wn     / m
    fl_n = flipped/m

    if   flip_channel=='left':
        stereo = np.vstack([fl_n, wn_n]).T
    elif flip_channel=='right':
        stereo = np.vstack([wn_n, fl_n]).T
    else:
        stereo = np.vstack([wn_n, wn_n]).T

    return stereo.astype(np.float32)

def save_wav_and_fft(stereo, freq, flip_channel):
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_path = f"stimuli/stim_{freq}Hz_{flip_channel}_{ts}.wav"
    write(wav_path, SAMPLE_RATE, (stereo*32767).astype(np.int16))

    ch   = stereo[:,1] if flip_channel=='right' else stereo[:,0]
    F    = fft(ch)
    mags = 20*np.log10(np.abs(F)+1e-12)
    phs  = np.angle(F)
    freqs= fftfreq(N_SAMPLES, d=1/SAMPLE_RATE)

    df = pd.DataFrame({
        'freq(Hz)': freqs[:N_SAMPLES//2],
        'dBFS':     mags[:N_SAMPLES//2],
        'phase':    phs[:N_SAMPLES//2]
    })
    df['flipped_band'] = df['freq(Hz)'].apply(
        lambda x: 'YES' if abs(x)>freq-BANDWIDTH and abs(x)<freq+BANDWIDTH else ''
    )
    df['channel'] = flip_channel
    df.to_csv(wav_path.replace('.wav','.csv'), index=False)

# =============================================================================
# 6) HEADPHONE CHECK (loops until passed or ESC)
# —————————————————————————————————————————————————————————————
while True:
    instr.text = (
        "HEADPHONE CHECK\n\n"
        "A 1 kHz tone will play in ONE ear only.\n"
        "After each, press LEFT or RIGHT.\n\n"
        "Press SPACE to begin or ESC to quit."
    )
    instr.draw(); win.flip()
    keys = event.waitKeys(keyList=['space','escape'])
    if 'escape' in keys:
        core.quit()

    hp_ok = True
    for _ in range(3):
        side = random.choice(['left','right'])
        tone = np.zeros((N_SAMPLES,2))
        t    = np.linspace(0, DURATION, N_SAMPLES)
        pure = np.sin(2*np.pi*1000*t)
        tone[:,0 if side=='left' else 1] = pure

        s = sound.Sound(value=tone, sampleRate=SAMPLE_RATE, stereo=True)
        s.play(); core.wait(DURATION)

        instr.text = "Which ear? (LEFT / RIGHT) — ESC to quit"
        instr.draw(); win.flip()
        k = event.waitKeys(keyList=['left','right','escape'])[0]
        if k=='escape':
            core.quit()
        if k!=side:
            hp_ok = False

    if hp_ok:
        break
    else:
        instr.text = "Headphone check FAILED.\n\nLet's try again."
        instr.draw(); win.flip()
        core.wait(2.0)

# =============================================================================
# 7) PURE-TONE HEARING TEST
# —————————————————————————————————————————————————————————————
instr.text = (
    "PURE-TONE TEST\n\n"
    "You will hear tones at various frequencies.\n"
    "Press Y if you hear it, N if not.\n\n"
    "Press SPACE to begin or ESC to quit."
)
instr.draw(); win.flip()
keys = event.waitKeys(keyList=['space','escape'])
if 'escape' in keys:
    core.quit()

pure_tones = [250,500,1000,2000,4000]
thresholds  = {}
for f in pure_tones:
    instr.text = f"{f} Hz: Hear it? (Y/N)"
    instr.draw(); win.flip()
    tone = np.sin(2*np.pi*f*np.linspace(0,DURATION,N_SAMPLES))
    stereo = np.vstack([tone,tone]).T * 0.1
    sd = sound.Sound(value=stereo, sampleRate=SAMPLE_RATE, stereo=True)
    sd.play(); core.wait(DURATION)
    k = event.waitKeys(keyList=['y','n','escape'])[0]
    if k=='escape':
        core.quit()
    thresholds[f] = k

# =============================================================================
# 8) PRACTICE BLOCK
# —————————————————————————————————————————————————————————————
instr.text = (
    "PRACTICE BLOCK\n\n"
    "You will hear exaggerated pitch illusions.\n"
    "Press SPACE to continue or ESC to quit."
)
instr.draw(); win.flip()
keys = event.waitKeys(keyList=['space','escape'])
if 'escape' in keys:
    core.quit()

for f in [400,800]:
    stim = create_dichotic_pitch(f,'right')
    sound.Sound(value=stim,sampleRate=SAMPLE_RATE,stereo=True).play()
    core.wait(DURATION)
    instr.text = "Did you hear a pitch? (Y/N)"
    instr.draw(); win.flip()
    k = event.waitKeys(keyList=['y','n','escape'])[0]
    if k=='escape':
        core.quit()
    core.wait(0.5)

# =============================================================================
# 9) MAIN DICHOTIC-PITCH EXPERIMENT
# —————————————————————————————————————————————————————————————
for block in range(1, NUM_BLOCKS+1):
    tests = random.sample([f for f in TEST_FREQS if f!=CONTROL_FREQ], 3)
    seq   = [CONTROL_FREQ] + tests + [CONTROL_FREQ]

    for trial, freq in enumerate(seq,1):
        typ  = 'Control' if freq==CONTROL_FREQ else 'Test'
        flip = random.choice(['left','right']) if typ=='Test' else 'none'
        stim = create_dichotic_pitch(freq, flip)

        if typ=='Test':
            save_wav_and_fft(stim, freq, flip)

        fix.draw(); win.flip(); core.wait(1.0)
        sound.Sound(value=stim,sampleRate=SAMPLE_RATE,stereo=True).play()
        core.wait(DURATION)

        instr.text = "Hear a pitch? (Y/N)"
        instr.draw(); win.flip()
        key,rt = event.waitKeys(keyList=['y','n','escape'], timeStamped=core.Clock())[0]
        if key=='escape':
            core.quit()

        instr.text = "Confidence (1–5)"
        instr.draw(); win.flip()
        conf,_ = event.waitKeys(keyList=[str(i) for i in range(1,6)]+['escape'], timeStamped=core.Clock())[0]
        if conf=='escape':
            core.quit()

        if key=='y':
            instr.text = "Which ear? (LEFT/RIGHT)"
            instr.draw(); win.flip()
            ear = event.waitKeys(keyList=['left','right','escape'])[0]
            if ear=='escape':
                core.quit()

            instr.text = "High or Low? (H/L)"
            instr.draw(); win.flip()
            pt  = event.waitKeys(keyList=['h','l','escape'])[0]
            if pt=='escape':
                core.quit()
        else:
            ear, pt = 'NA','NA'

        corr   = ('yes' if (typ=='Control' and key=='n')
                       or (typ=='Test'    and key=='y') else 'no')
        ear_ok = ('yes' if flip==ear else 'no') if flip in ['left','right'] else 'NA'

        row = [
            str(block), str(trial), typ, str(freq),
            key, conf, f"{rt:.3f}",
            ear, ('high' if pt=='h' else 'low' if pt=='l' else 'NA'),
            corr, flip, ear_ok
        ]
        data_file.write(','.join(row)+'\n')
        data_file.flush()
        core.wait(0.5)

    if block<NUM_BLOCKS:
        instr.text = "Short break—press SPACE to continue or ESC to quit."
        instr.draw(); win.flip()
        k = event.waitKeys(keyList=['space','escape'])[0]
        if k=='escape':
            core.quit()

# =============================================================================
# 10) CLEANUP
# —————————————————————————————————————————————————————————————
data_file.close()
instr.text = "Experiment complete.\n\nThank you!"
instr.draw(); win.flip()
core.wait(3.0)
win.close(); core.quit()
