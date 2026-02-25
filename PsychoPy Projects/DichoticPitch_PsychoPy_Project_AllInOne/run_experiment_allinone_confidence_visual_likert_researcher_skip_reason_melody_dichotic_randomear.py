import os, time, json, csv
import numpy as np
from psychopy import visual, core, event, gui, sound, data
from psychopy.constants import STARTED
import soundfile as sf
# ============================================================
# All-in-one helpers (stimulus generation + IO)
# ============================================================

def _rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x * x)) + 1e-12)

def _normalize_peak(x: np.ndarray, peak: float = 0.99) -> np.ndarray:
    m = float(np.max(np.abs(x)) + 1e-12)
    return (x / m) * peak

def make_dichotic_stim(params: dict, fs: int, rng: np.random.Generator) -> np.ndarray:
    """Create a simple dichotic pitch-style stereo stimulus.

    Strategy:
      - Start with identical white noise in both ears
      - In a narrow band around carrier_hz, apply a phase rotation (or inversion) to RIGHT ear only
      - Sham trials keep ears identical

    Returns:
      y: shape (n_samples, 2) float32 in [-1, 1]
    """
    dur = float(params.get("dur", 0.75))
    carrier = float(params.get("carrier_hz", 250.0))
    phase_offset = float(params.get("phase_offset_rad", np.pi))  # default inversion-ish
    is_sham = bool(params.get("is_sham", False))

    # Narrow-band width (Hz). Keep conservative to reduce spectral splatter.
    bw = float(params.get("bandwidth_hz", 50.0))
    n = int(round(dur * fs))
    n = max(n, 1)

    # Base noise (same seed already handled by rng passed in)
    noise = rng.standard_normal(n).astype(np.float64)

    # FFT
    X = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    # Band mask
    half = bw / 2.0
    band = (freqs >= (carrier - half)) & (freqs <= (carrier + half))

    # Left ear = original
    XL = X.copy()

    # Right ear: apply phase shift in the band only (or identical for sham)
    XR = X.copy()
    if (not is_sham) and np.any(band):
        XR[band] *= np.exp(1j * phase_offset)

    # Back to time domain
    left = np.fft.irfft(XL, n=n)
    right = np.fft.irfft(XR, n=n)

    # Gentle ramp to avoid clicks
    ramp_ms = float(params.get("ramp_ms", 10.0))
    ramp = int(round((ramp_ms / 1000.0) * fs))
    if ramp > 1 and ramp * 2 < n:
        w = np.ones(n, dtype=np.float64)
        r = np.linspace(0, 1, ramp, endpoint=False)
        w[:ramp] = r
        w[-ramp:] = r[::-1]
        left *= w
        right *= w

    y = np.column_stack([left, right])

    # Safe peak normalize
    y = _normalize_peak(y, peak=float(params.get("peak", 0.99))).astype(np.float32)
    return y

def make_melody(params: dict, fs: int, rng: np.random.Generator) -> np.ndarray:
    """Create a melody where each note is generated using the *same* dichotic-pitch
    method as the main dichotic pitch section.

    Each note is a short dichotic-pitch-style noise burst with a phase-rotated
    narrow band around that note's carrier frequency.

    Returns:
      y: (n_samples, 2) float32
    """
    pitch_seq = [float(x) for x in params.get("pitch_seq_hz", [440.0])]
    note_dur = float(params.get("note_dur", 0.35))
    gap = float(params.get("gap", 0.05))

    # Match dichotic pitch defaults unless explicitly overridden
    bw = float(params.get("bandwidth_hz", 50.0))
    phase_offset = float(params.get("phase_offset_rad", np.pi))
    ramp_ms = float(params.get("ramp_ms", 10.0))

    chunks = []
    n_gap = int(round(gap * fs))

    for f0 in pitch_seq:
        note_params = {
            "dur": note_dur,
            "carrier_hz": f0,
            "phase_offset_rad": phase_offset,
            "is_sham": False,
            "bandwidth_hz": bw,
            "ramp_ms": ramp_ms,
            "peak": 0.99,
        }
        y_note = make_dichotic_stim(note_params, fs, rng)  # (n,2)
        chunks.append(y_note.astype(np.float64))
        if n_gap > 0:
            chunks.append(np.zeros((n_gap, 2), dtype=np.float64))

    if chunks:
        y = np.vstack(chunks)
    else:
        y = np.zeros((int(round(0.5 * fs)), 2), dtype=np.float64)

    # Normalize overall peak safely
    y = _normalize_peak(y, peak=0.99).astype(np.float32)
    return y

def save_wav_and_fft(y: np.ndarray, fs: int, wav_path: str, fft_path: str,
                     run_id: str = "", stim_id: str = "", stim_type: str = "",
                     center_freq_hz: float | None = None) -> None:
    """Save stereo WAV and write a basic FFT CSV (Frequency, Loudness dBFS, Phase).

    FFT is computed on a mono mix (L+R)/2 for compact reporting.
    """
    y = np.asarray(y)
    sf.write(wav_path, y, fs, subtype="PCM_16")

    mono = y.mean(axis=1) if (y.ndim == 2 and y.shape[1] == 2) else y.astype(np.float64)
    mono = mono.astype(np.float64)

    n = len(mono)
    if n < 2:
        return

    X = np.fft.rfft(mono)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    mag = np.abs(X)
    # Convert magnitude to dBFS-ish scale relative to 1.0 full-scale sine
    # (This is an approximation but stable for comparing bins)
    mag_db = 20.0 * np.log10(mag / (n / 2.0) + 1e-12)
    phase = np.angle(X)

    # Flag bins near center frequency if provided
    flag = np.zeros_like(freqs, dtype=int)
    if center_freq_hz is not None and np.isfinite(center_freq_hz):
        bw = 25.0
        flag = ((freqs >= center_freq_hz - bw) & (freqs <= center_freq_hz + bw)).astype(int)

    with open(fft_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run_id","stim_id","stim_type","Frequency_Hz","Loudness_dBFS","Phase_rad","NearCenterBand"])
        for fr, db, ph, fl in zip(freqs, mag_db, phase, flag):
            w.writerow([run_id, stim_id, stim_type, float(fr), float(db), float(ph), int(fl)])

def write_manifest(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


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

def show(msg, wait_key=False, keys=("space",)):
    """Display text on screen. Prefer using ask() to collect key responses."""
    txt.text = msg
    txt.draw()
    win.flip()
    if wait_key:
        _ = ask(msg, keyList=list(keys))

def ask(msg, keyList=("space",), timeStamped=False):
    """Show msg and wait for a single keypress (always includes ESCAPE).

    Returns:
      if timeStamped=False: key (str)
      if timeStamped=True : (key, rt_seconds)
    """
    txt.text = msg
    txt.draw()
    win.flip()

    # use a clock so RT is relative to prompt onset
    clock = core.Clock()
    clock.reset()

    keys = event.waitKeys(
        keyList=list(keyList) + ["escape"],
        timeStamped=clock if timeStamped else False
    )

    k = keys[0]
    if timeStamped:
        key, rt = k
    else:
        key, rt = k, None

    if key == "escape":
        core.quit()

    return (key, rt) if timeStamped else key


def wait_space_or_shift_s(prompt):
    """Show prompt, wait for SPACE (continue) or researcher-only Shift+S (skip)."""
    txt.text = prompt
    txt.draw()
    win.flip()
    keys = event.waitKeys(keyList=["space", "s", "escape"], modifiers=True)
    key, mods = keys[0]
    if key == "escape":
        core.quit()
    is_shift_s = (key == "s") and bool(mods.get("shift", False))
    return key, is_shift_s

def text_entry(prompt, allow_empty=False, max_len=120):
    """Simple on-screen text entry. RETURN to confirm, BACKSPACE to delete."""
    entry = ""
    while True:
        txt.text = prompt + "\n\n" + entry + "\n\n(ENTER=done, BACKSPACE=delete, ESC=quit)"
        txt.draw()
        win.flip()
        k = event.waitKeys()[0]
        if k == "escape":
            core.quit()
        if k in ("return", "num_enter"):
            if allow_empty or entry.strip():
                return entry.strip()
            core.wait(0.15)
            continue
        if k == "backspace":
            entry = entry[:-1]
            continue
        if k == "space":
            if len(entry) < max_len:
                entry += " "
            continue
        if isinstance(k, str) and len(k) == 1 and len(entry) < max_len:
            entry += k

# ----------------------------
# Phase A: File Acquisition
# ----------------------------
fs = 48000
base_dur = 0.75
rng = np.random.default_rng()

show("Phase A: Generating stimuli files...\n(Press SPACE to start)", wait_key=True)

items = []
melodies = []

# ----------------------------
# Combined response + confidence (single screen, labeled scale)
# ----------------------------
CONF_LABELS = {
    "1": "Not sure",
    "2": "Slightly sure",
    "3": "Moderately sure",
    "4": "Very sure",
    "5": "Extremely sure",
}

def ask_with_confidence(main_msg, main_keys, conf_labels=CONF_LABELS):
    """Two-screen response collection:
      Screen 1: main response (e.g., Y/N or 0-6)
      Screen 2: visual Likert confidence scale (1-5) with labels

    Returns: (main_key, main_rt, conf_key, conf_rt)
      - main_rt: seconds from main prompt onset to main key
      - conf_rt: seconds from confidence screen onset to confidence key
    """
    # ---------- Screen 1: main response ----------
    txt.text = main_msg
    txt.draw()
    win.flip()

    clock_main = core.Clock()
    clock_main.reset()
    key, t = event.waitKeys(
        keyList=list(main_keys) + ["escape"],
        timeStamped=clock_main
    )[0]
    if key == "escape":
        core.quit()
    main_rt = t

    # ---------- Screen 2: confidence (visual Likert) ----------
    # Build ordered labels 1..5
    ordered = [(str(i), conf_labels.get(str(i), "")) for i in range(1, 6)]

    # Layout (units="height")
    prompt_y = 0.35
    boxes_y  = 0.02
    label_y  = -0.18
    box_w    = 0.14
    box_h    = 0.12
    gap      = 0.04
    total_w  = 5 * box_w + 4 * gap
    start_x  = -total_w / 2 + box_w / 2

    prompt = visual.TextStim(
        win,
        text="Rate your confidence (press 1–5):",
        height=0.05,
        color=(1, 1, 1),
        pos=(0, prompt_y),
        wrapWidth=1.3
    )

    rects = []
    nums  = []
    labs  = []

    for idx, (k, lab) in enumerate(ordered):
        x = start_x + idx * (box_w + gap)

        r = visual.Rect(
            win,
            width=box_w,
            height=box_h,
            pos=(x, boxes_y),
            lineColor=(1, 1, 1),
            fillColor=None
        )
        rects.append(r)

        n = visual.TextStim(
            win,
            text=k,
            height=0.05,
            color=(1, 1, 1),
            pos=(x, boxes_y)
        )
        nums.append(n)

        l = visual.TextStim(
            win,
            text=lab,
            height=0.03,
            color=(1, 1, 1),
            pos=(x, label_y),
            wrapWidth=box_w + 0.02
        )
        labs.append(l)

    # Draw confidence screen
    prompt.draw()
    for r, n, l in zip(rects, nums, labs):
        r.draw(); n.draw(); l.draw()
    win.flip()

    clock_conf = core.Clock()
    clock_conf.reset()
    conf_key, conf_t = event.waitKeys(
        keyList=["1","2","3","4","5","escape"],
        timeStamped=clock_conf
    )[0]
    if conf_key == "escape":
        core.quit()
    conf_rt = conf_t

    # Brief highlight feedback (kept subtle; does not reveal sham/stim)
    try:
        sel = int(conf_key) - 1
        if 0 <= sel < 5:
            rects[sel].fillColor = (0.35, 0.35, 0.35)
            prompt.draw()
            for r, n, l in zip(rects, nums, labs):
                r.draw(); n.draw(); l.draw()
            win.flip()
            core.wait(0.12)
    except Exception:
        pass

    return key, main_rt, conf_key, conf_rt


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
# --- Phase C entry (hidden researcher-only skip: Shift+S) ---
_key, _shift_s = wait_space_or_shift_s(
    "Phase C: Hearing test\n""You\'ll hear tones in LEFT or RIGHT ear.\n""Press Y if heard, N if not.\n""Press SPACE to start."
)

ht_skipped = False
ht_skip_reason = ""

if _shift_s:
    # researcher confirmation (not shown in the regular instructions above)
    conf = ask("Researcher: Skip the hearing test?\nY = Yes (skip)\nN = No (run hearing test)", keyList=("y","n"))
    if conf == "y":
        ht_skipped = True
        ht_skip_reason = text_entry("Reason for skipping hearing test:", allow_empty=False)

thisExp.addData("ht_skipped", ht_skipped)
thisExp.addData("ht_skip_reason", ht_skip_reason)
thisExp.nextEntry()

if not ht_skipped:
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

                resp = ask(f"Hearing test: {ear} ear @ {freq} Hz\nHeard it? (Y/N)", keyList=("y","n"))

                if resp == "y":
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
        w.writerow(["ear","freq_hz","threshold_amp","skipped","reason"])
        for ear in ["L","R"]:
            for freq in test_freqs:
                w.writerow([ear, freq, audiogram[ear][freq], False, ""])

else:
    # If skipped, use neutral thresholds so later compensation doesn't break
    test_freqs = [250, 500, 1000, 2000, 4000, 8000]
    audiogram = {"L": {f: levels[0] for f in test_freqs}, "R": {f: levels[0] for f in test_freqs}}

    # Save audiogram CSV noting skip + reason
    with open(audiogram_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ear","freq_hz","threshold_amp","skipped","reason"])
        for ear in ["L","R"]:
            for freq in test_freqs:
                w.writerow([ear, freq, audiogram[ear][freq], True, ht_skip_reason])

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
        w.writerow(["run_id","participant","trialN","stimId","stimType","stimFile","targetEar","heardEar","correct","rt","confidence","conf_rt"])

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

    # Load wav so we can optionally swap channels for random ear presentation
    y, fsi = sf.read(stimFile, dtype="float32", always_2d=True)
    if fsi != fs:
        # assume consistent fs across files; if not, PsychoPy will still play but timing may drift
        pass

    # Randomize perceived-ear presentation by swapping channels on the fly (keeps participant blind)
    target_ear = rng.choice(["L","R"])  # which ear should carry the phase-altered channel
    if target_ear == "L":
        y = y[:, ::-1]  # swap L/R

    s = sound.Sound(value=y, sampleRate=fsi, stereo=True)
    s.setVolume(hp_volume)
    s.play()
    core.wait(s.getDuration())
    s.stop()

    # response: single forced-choice about ear (plus "N" for no tone)
    (resp_key, rt, conf_key, conf_rt) = ask_with_confidence(
        f"Trial {trialN+1}/{len(trial_rows)}
Where did you hear the pitch?
(L=Left  R=Right  N=No tone)",
        main_keys=("l","r","n")
    )

    heard_ear = resp_key.upper()
    # Determine correctness:
    # - For sham trials, correct = N
    # - For stim trials, correct = target ear (L/R)
    if stimType.lower() == "sham":
        correct = (heard_ear == "N")
    else:
        correct = (heard_ear == target_ear)


    with open(responses_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([run_id, expInfo.get("participant",""), trialN, stimId, stimType, stimFile, target_ear, heard_ear, int(correct), rt, conf_key, conf_rt])

    thisExp.addData("phase", "E_main")
    thisExp.addData("trialN", trialN)
    thisExp.addData("stimId", stimId)
    thisExp.addData("stimType", stimType)
    thisExp.addData("stimFile", stimFile)
    thisExp.addData("targetEar", "")
    thisExp.addData("heardEar", "")
    thisExp.addData("correct", "")
    thisExp.addData("targetEar", target_ear)
    thisExp.addData("heardEar", heard_ear)
    thisExp.addData("correct", int(correct))
    thisExp.addData("resp", resp_key)
    thisExp.addData("rt", rt)
    thisExp.addData("confidence", int(conf_key))
    thisExp.addData("conf_rt", conf_rt)
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

    (resp_key, rt, conf_key, conf_rt) = ask_with_confidence(
        f"Melody {mN+1}/{len(mel_rows)}\nHow many pitches did you hear? (0-6)",
        main_keys=("0","1","2","3","4","5","6")
)


    with open(responses_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([run_id, expInfo.get("participant",""), f"mel_{mN}", stimId, "melody", stimFile, "", "", "", rt, conf_key, conf_rt])

    thisExp.addData("phase", "E_melody")
    thisExp.addData("trialN", f"mel_{mN}")
    thisExp.addData("stimId", stimId)
    thisExp.addData("stimType", "melody")
    thisExp.addData("stimFile", stimFile)
    thisExp.addData("targetEar", "")
    thisExp.addData("heardEar", "")
    thisExp.addData("correct", "")
    thisExp.addData("targetEar", target_ear)
    thisExp.addData("heardEar", heard_ear)
    thisExp.addData("correct", int(correct))
    thisExp.addData("melody_count", int(resp_key))
    thisExp.addData("rt", rt)
    thisExp.addData("confidence", int(conf_key))
    thisExp.addData("conf_rt", conf_rt)
    thisExp.nextEntry()

show("Done!\nThank you.\nPress SPACE to exit.", True)
thisExp.saveAsWideText(thisExp.dataFileName + ".csv")
thisExp.abort()  # close handler
win.close()
core.quit()
