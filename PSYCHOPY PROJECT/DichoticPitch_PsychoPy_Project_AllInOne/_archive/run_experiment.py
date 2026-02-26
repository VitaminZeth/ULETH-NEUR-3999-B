import os, time, json, csv
import numpy as np
from psychopy import visual, core, event, gui, sound, data
from psychopy.constants import STARTED
import soundfile as sf

from stimuli import (
    make_dichotic_stim, make_melody, save_wav_and_fft, write_manifest
)

# ----------------------------
# Experiment info / UI
# ----------------------------
expInfo = {"participant": ""}
dlg = gui.DlgFromDict(expInfo, title="Dichotic Pitch Experiment")
if not dlg.OK:
    core.quit()

# ----------------------------
# Folder structure per run
# ----------------------------
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)

root_out = os.path.join(_thisDir, "OUTPUT")
os.makedirs(root_out, exist_ok=True)

run_id = time.strftime("%Y%m%d_%H%M%S")
run_dir = os.path.join(root_out, f"RUN_{run_id}")
os.makedirs(run_dir, exist_ok=True)

phaseA_dir  = os.path.join(run_dir, "PHASEA_FileAcquisition")
phaseA_wav  = os.path.join(phaseA_dir, "WAV")
phaseA_fft  = os.path.join(phaseA_dir, "FFT_CSV")
phaseA_logs = os.path.join(phaseA_dir, "LOGS")

phaseC_dir  = os.path.join(run_dir, "PHASEC_HearingTest")
phaseC_logs = os.path.join(phaseC_dir, "LOGS")

phaseD_dir  = os.path.join(run_dir, "PHASED_Preparation")
phaseD_wav  = os.path.join(phaseD_dir, "WAV_COMP")
phaseD_logs = os.path.join(phaseD_dir, "LOGS")

phaseE_dir  = os.path.join(run_dir, "PHASEE_Trials")
phaseE_logs = os.path.join(phaseE_dir, "LOGS")

for p in [phaseA_wav, phaseA_fft, phaseA_logs, phaseC_logs, phaseD_wav, phaseD_logs, phaseE_logs]:
    os.makedirs(p, exist_ok=True)

manifest_path     = os.path.join(phaseA_logs, "manifest.json")
stim_features_csv = os.path.join(phaseA_logs, "stim_features.csv")
trial_list_csv    = os.path.join(phaseA_logs, "trial_list.csv")
mel_list_csv      = os.path.join(phaseA_logs, "melody_list.csv")
responses_csv     = os.path.join(phaseE_logs, "responses.csv")
audiogram_csv     = os.path.join(phaseC_logs, "audiogram.csv")
prep_log_json     = os.path.join(phaseD_logs, "prep_gains.json")

# PsychoPy experiment handler for main .csv
thisExp = data.ExperimentHandler(
    name="DichoticPitch",
    extraInfo={**expInfo, "run_id": run_id},
    dataFileName=os.path.join(run_dir, f"data_{expInfo['participant']}_{run_id}")
)

# ----------------------------
# Window
# ----------------------------
win = visual.Window(fullscr=False, size=(1024, 768), units="height", color=(0,0,0))
txt = visual.TextStim(win, text="", height=0.04, wrapWidth=1.2, color=(1,1,1))

def show(msg, wait_key=True, keys=("space",)):
    txt.text = msg
    txt.draw()
    win.flip()
    if wait_key:
        event.waitKeys(keyList=list(keys) + ["escape"])
        if "escape" in event.getKeys(["escape"]):
            core.quit()

# ----------------------------
# Phase A: File Acquisition
# ----------------------------
fs = 48000
base_dur = 0.75
rng = np.random.default_rng()

show("Phase A: Generating stimuli files...\n(Press SPACE to start)", wait_key=True)

items = []
melodies = []

def gen_one(params, stim_type):
    stim_id = params["id"]
    if stim_type == "melody":
        y = make_melody(params, fs, rng)
        center = None
    else:
        y = make_dichotic_stim(params, fs, rng)
        center = params.get("carrier_hz", None)

    wav_path = os.path.join(phaseA_wav, f"{run_id}_{stim_id}.wav")
    fft_path = os.path.join(phaseA_fft, f"{run_id}_{stim_id}_FFT.csv")

    save_wav_and_fft(y, fs, wav_path, fft_path, run_id=run_id, stim_id=stim_id, stim_type=stim_type, center_freq_hz=center)

    entry = {
        "run_id": run_id,
        "stimId": stim_id,
        "stimType": stim_type,
        "centerFreqHz": center,
        "wavPath": wav_path,
        "fftCsvPath": fft_path,
        "params": params
    }
    return entry

# 3 shams
for i in range(3):
    params = {
        "id": f"sham_{i+1}",
        "dur": base_dur,
        "carrier_hz": float(rng.choice([200, 250, 315, 400, 500, 630])),
        "phase_offset_rad": 0.0,
        "is_sham": True
    }
    items.append(gen_one(params, "sham"))

# 5 stimuli
for i in range(5):
    params = {
        "id": f"stim_{i+1}",
        "dur": base_dur,
        "carrier_hz": float(rng.choice([200, 250, 315, 400, 500, 630])),
        "phase_offset_rad": float(rng.uniform(-np.pi, np.pi)),
        "is_sham": False
    }
    items.append(gen_one(params, "stim"))

# 2 melodies, each with 6 pitches
pitch_pool = np.array([220, 247, 277, 311, 349, 392, 440, 494], dtype=float)
for i in range(2):
    params = {
        "id": f"melody_{i+1}",
        "pitch_seq_hz": rng.choice(pitch_pool, size=6, replace=True).tolist(),
        "note_dur": 0.35,
        "gap": 0.05,
        "phase_offset_rad": float(rng.uniform(-np.pi, np.pi)),
    }
    melodies.append(gen_one(params, "melody"))

# Write Phase A logs
write_manifest(manifest_path, {"run_id": run_id, "fs": fs, "items": items, "melodies": melodies})

with open(stim_features_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["run_id","stimId","stimType","centerFreqHz","wavPath","fftCsvPath"])
    for it in items + melodies:
        w.writerow([it["run_id"], it["stimId"], it["stimType"], it["centerFreqHz"], it["wavPath"], it["fftCsvPath"]])

with open(trial_list_csv, "w", encoding="utf-8") as f:
    f.write("stimFile,stimType,stimId,centerFreqHz,fftCsvPath\n")
    for it in items:
        f.write(f"{it['wavPath']},{it['stimType']},{it['stimId']},{it['centerFreqHz']},{it['fftCsvPath']}\n")

with open(mel_list_csv, "w", encoding="utf-8") as f:
    f.write("stimFile,stimType,stimId,centerFreqHz,fftCsvPath\n")
    for it in melodies:
        f.write(f"{it['wavPath']},{it['stimType']},{it['stimId']},{it['centerFreqHz']},{it['fftCsvPath']}\n")

show("Phase A complete.\nFiles + FFT CSVs exported.\nPress SPACE to continue.", True)

# ----------------------------
# Phase B: Calibration (volume + L/R orientation)
# ----------------------------
hp_volume = 0.3
fs_cal = 48000

# --- Reference test tone (440 Hz) ---
# Press 't' to toggle the tone ON/OFF while you adjust volume.
tone_on = False
tone_freq = 440.0
tone_dur = 1.0  # seconds

t_tone = np.arange(int(fs_cal * tone_dur)) / fs_cal
tone = (0.2 * np.sin(2 * np.pi * tone_freq * t_tone)).astype(np.float32)
tone_stereo = np.column_stack([tone, tone]).astype(np.float32)
ref_tone = sound.Sound(value=tone_stereo, sampleRate=fs_cal, stereo=True)

# We'll loop the tone manually (more compatible across PsychoPy sound backends)
tone_clock = core.Clock()
tone_clock.reset()

# --- L/R orientation beeps ---
t = np.arange(int(fs_cal*0.25))/fs_cal
beep = (0.2*np.sin(2*np.pi*800*t)).astype(np.float32)
left_beep = np.column_stack([beep, np.zeros_like(beep)])
right_beep = np.column_stack([np.zeros_like(beep), beep])

snd_left = sound.Sound(value=left_beep, sampleRate=fs_cal, stereo=True)
snd_right = sound.Sound(value=right_beep, sampleRate=fs_cal, stereo=True)

show(
    "Phase B: Calibration\n"
    "UP/DOWN = volume\n"
    "T = toggle 440 Hz reference tone\n"
    "LEFT = play left beep\n"
    "RIGHT = play right beep\n"
    "SPACE = continue",
    False
)

while True:
    # Keep the reference tone looping if it's ON
    if tone_on:
        ref_tone.setVolume(hp_volume)
        # If the tone finished (or isn't started), restart every tone_dur seconds.
        if (ref_tone.status != STARTED) or (tone_clock.getTime() >= tone_dur):
            ref_tone.stop()
            ref_tone.play()
            tone_clock.reset()

    for k in event.getKeys():
        if k == "escape":
            core.quit()
        if k == "up":
            hp_volume = min(1.0, hp_volume + 0.02)
        elif k == "down":
            hp_volume = max(0.0, hp_volume - 0.02)
        elif k == "t":
            tone_on = not tone_on
            if not tone_on:
                ref_tone.stop()
            else:
                ref_tone.setVolume(hp_volume)
                ref_tone.play()
                tone_clock.reset()
        elif k == "left":
            snd_left.setVolume(hp_volume); snd_left.play()
        elif k == "right":
            snd_right.setVolume(hp_volume); snd_right.play()
        elif k == "space":
            # stop tone before moving on
            ref_tone.stop()
            txt.text = ""
            win.flip()
            break
    else:
        txt.text = (
            f"Phase B: Calibration\n"
            f"Volume: {hp_volume:.2f}\n"
            f"440 Hz tone: {'ON' if tone_on else 'OFF'} (press T)\n"
            "UP/DOWN adjust, LEFT/RIGHT test, SPACE continue"
        )
        txt.draw(); win.flip()
        core.wait(0.01)
        continue
    break

thisExp.addData("hp_volume", hp_volume)
thisExp.addData("cal_refTone440_onExit", tone_on)
thisExp.nextEntry()

# ----------------------------
# Phase C: Hearing test (simple threshold per ear)
# ----------------------------
show("Phase C: Hearing test\nYou'll hear tones in LEFT or RIGHT ear.\nPress Y if heard, N if not.\nPress SPACE to start.", True)

test_freqs = [250, 500, 1000, 2000, 4000, 8000]
levels = [0.25, 0.18, 0.12, 0.08, 0.05, 0.03, 0.02]  # linear amp levels
audiogram = {"L": {}, "R": {}}

def make_ear_tone(freq, amp, ear, fs=48000, dur=0.5):
    t = np.arange(int(fs*dur))/fs
    tone = (amp * np.sin(2*np.pi*freq*t)).astype(np.float32)
    if ear == "L":
        return np.column_stack([tone, np.zeros_like(tone)])
    else:
        return np.column_stack([np.zeros_like(tone), tone])

for ear in ["L", "R"]:
    for freq in test_freqs:
        level_i = 0
        while True:
            amp = levels[level_i]
            arr = make_ear_tone(freq, amp, ear, fs=fs, dur=0.5)
            s = sound.Sound(value=arr, sampleRate=fs, stereo=True)
            s.setVolume(hp_volume)
            s.play()
            core.wait(0.55)

            show(f"Hearing test: {ear} ear @ {freq} Hz\nHeard it? (Y/N)", True, keys=("y","n"))

            resp = event.getKeys(["y","n","escape"])
            if "escape" in resp:
                core.quit()

            if "y" in resp:
                if level_i < len(levels) - 1:
                    level_i += 1  # go softer
                    continue
                else:
                    audiogram[ear][freq] = levels[-1]
                    break
            else:  # "n"
                audiogram[ear][freq] = None if level_i == 0 else levels[level_i-1]
                break

        thisExp.addData("ht_ear", ear)
        thisExp.addData("ht_freq", freq)
        thisExp.addData("ht_threshold_amp", audiogram[ear][freq])
        thisExp.nextEntry()

# Save audiogram CSV
with open(audiogram_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["ear","freq_hz","threshold_amp"])
    for ear in ["L","R"]:
        for freq in test_freqs:
            w.writerow([ear, freq, audiogram[ear][freq]])

show("Phase C complete.\nPress SPACE to continue.", True)

# ----------------------------
# Phase D: Preparation (apply volume + audiogram gains to create compensated WAVs)
# ----------------------------
def ear_gain_map(ear_dict):
    gains = {}
    for freq, thr in ear_dict.items():
        if thr is None:
            gains[freq] = 2.5
        else:
            gains[freq] = float(np.clip(levels[0] / max(thr, 1e-6), 0.8, 2.5))
    return gains

gL = ear_gain_map(audiogram["L"])
gR = ear_gain_map(audiogram["R"])
L_gain = float(np.median(list(gL.values())))
R_gain = float(np.median(list(gR.values())))

def compensate_wav(in_path, out_path):
    y, fsi = sf.read(in_path, dtype="float32", always_2d=True)
    if fsi != fs:
        # keep simple: assume fs matches
        pass
    y[:,0] *= L_gain
    y[:,1] *= R_gain
    y = np.clip(y, -1.0, 1.0)
    sf.write(out_path, y, fs, subtype="PCM_16")
    return out_path

# Build compensated lists
trial_list_comp = os.path.join(phaseD_logs, "trial_list_COMP.csv")
mel_list_comp   = os.path.join(phaseD_logs, "melody_list_COMP.csv")

with open(trial_list_comp, "w", encoding="utf-8") as f:
    f.write("stimFile,stimType,stimId\n")
    for it in items:
        base = os.path.splitext(os.path.basename(it["wavPath"]))[0]
        out_path = os.path.join(phaseD_wav, base + "_COMP.wav")
        compensate_wav(it["wavPath"], out_path)
        f.write(f"{out_path},{it['stimType']},{it['stimId']}\n")

with open(mel_list_comp, "w", encoding="utf-8") as f:
    f.write("stimFile,stimType,stimId\n")
    for it in melodies:
        base = os.path.splitext(os.path.basename(it["wavPath"]))[0]
        out_path = os.path.join(phaseD_wav, base + "_COMP.wav")
        compensate_wav(it["wavPath"], out_path)
        f.write(f"{out_path},{it['stimType']},{it['stimId']}\n")

with open(prep_log_json, "w", encoding="utf-8") as f:
    json.dump({"L_gain": L_gain, "R_gain": R_gain, "hp_volume": hp_volume, "gL": gL, "gR": gR}, f, indent=2)

show("Phase D complete (compensated WAVs created).\nPress SPACE to continue.", True)

# ----------------------------
# Phase E: Trials (random playback) + response log CSV
# ----------------------------
if not os.path.exists(responses_csv):
    with open(responses_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run_id","participant","trialN","stimId","stimType","stimFile","response","rt"])

# Load trials
trial_rows = []
with open(trial_list_comp, "r", encoding="utf-8") as f:
    rdr = csv.DictReader(f)
    for row in rdr:
        trial_rows.append(row)

rng.shuffle(trial_rows)

show("Phase E: Main trials\nPress Y if you heard the pitch, N if not.\nPress SPACE to start.", True)

for trialN, row in enumerate(trial_rows):
    stimFile = row["stimFile"]
    stimType = row["stimType"]
    stimId   = row["stimId"]

    s = sound.Sound(stimFile)
    s.setVolume(hp_volume)  # global user volume (already compensated too; keep)
    s.play()
    core.wait(s.getDuration())
    s.stop()

    # response
    t0 = core.getTime()
    show(f"Trial {trialN+1}/{len(trial_rows)}\nStim: {stimType} ({stimId})\nHeard it? (Y/N)", True, keys=("y","n"))
    keys = event.getKeys(["y","n","escape"], timeStamped=True)
    if any(k[0]=="escape" for k in keys):
        core.quit()

    resp_key, resp_time = keys[-1]
    rt = resp_time - t0

    with open(responses_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([run_id, expInfo.get("participant",""), trialN, stimId, stimType, stimFile, resp_key, rt])

    thisExp.addData("phase", "E_main")
    thisExp.addData("trialN", trialN)
    thisExp.addData("stimId", stimId)
    thisExp.addData("stimType", stimType)
    thisExp.addData("stimFile", stimFile)
    thisExp.addData("resp", resp_key)
    thisExp.addData("rt", rt)
    thisExp.nextEntry()

# Melodic trials
mel_rows = []
with open(mel_list_comp, "r", encoding="utf-8") as f:
    rdr = csv.DictReader(f)
    for row in rdr:
        mel_rows.append(row)

rng.shuffle(mel_rows)

show("Melodic section\nYou will hear a short melody (6 pitches).\nPress 0-6 for how many pitches you heard.\nPress SPACE to start.", True)

for mN, row in enumerate(mel_rows):
    stimFile = row["stimFile"]
    stimId   = row["stimId"]

    s = sound.Sound(stimFile)
    s.setVolume(hp_volume)
    s.play()
    core.wait(s.getDuration())
    s.stop()

    t0 = core.getTime()
    show(f"Melody {mN+1}/{len(mel_rows)} ({stimId})\nHow many pitches did you hear? (0-6)", False)
    keys = event.waitKeys(keyList=["0","1","2","3","4","5","6","escape"], timeStamped=True)
    if keys[0][0] == "escape":
        core.quit()
    resp_key, resp_time = keys[0]
    rt = resp_time - t0

    with open(responses_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([run_id, expInfo.get("participant",""), f"mel_{mN}", stimId, "melody", stimFile, resp_key, rt])

    thisExp.addData("phase", "E_melody")
    thisExp.addData("trialN", f"mel_{mN}")
    thisExp.addData("stimId", stimId)
    thisExp.addData("stimType", "melody")
    thisExp.addData("stimFile", stimFile)
    thisExp.addData("melody_count", int(resp_key))
    thisExp.addData("rt", rt)
    thisExp.nextEntry()

show("Done!\nThank you.\nPress SPACE to exit.", True)
thisExp.saveAsWideText(thisExp.dataFileName + ".csv")
thisExp.abort()  # close handler
win.close()
core.quit()
