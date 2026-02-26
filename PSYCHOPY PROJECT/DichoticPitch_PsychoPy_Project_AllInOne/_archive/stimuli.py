import os, csv, json
import numpy as np
import soundfile as sf

def normalize(x, peak=0.98):
    m = float(np.max(np.abs(x)))
    if m < 1e-12:
        return x
    return (x / m) * peak

def compute_fft_csv(stereo, fs, out_csv_path, run_id=None, stim_id=None, stim_type=None, center_freq_hz=None):
    stereo = np.asarray(stereo)
    if stereo.ndim != 2 or stereo.shape[1] != 2:
        raise ValueError("compute_fft_csv expects stereo array (nSamples, 2)")

    xL = stereo[:, 0].astype(np.float64)
    xR = stereo[:, 1].astype(np.float64)

    XL = np.fft.rfft(xL)
    XR = np.fft.rfft(xR)
    freqs = np.fft.rfftfreq(len(xL), d=1.0/fs)

    eps = 1e-12
    magL = 20.0 * np.log10((np.abs(XL) / (len(xL)/2.0)) + eps)
    magR = 20.0 * np.log10((np.abs(XR) / (len(xR)/2.0)) + eps)
    phL = np.angle(XL)
    phR = np.angle(XR)

    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if run_id is not None:
            w.writerow(["meta", "run_id", run_id])
        if stim_id is not None:
            w.writerow(["meta", "stim_id", stim_id])
        if stim_type is not None:
            w.writerow(["meta", "stim_type", stim_type])
        w.writerow(["meta", "fs", fs])
        if center_freq_hz is not None:
            w.writerow(["meta", "center_freq_hz", center_freq_hz])
        w.writerow(["Frequency_Hz", "Mag_L_dBFS", "Mag_R_dBFS", "Phase_L_rad", "Phase_R_rad"])
        for i in range(len(freqs)):
            w.writerow([float(freqs[i]), float(magL[i]), float(magR[i]), float(phL[i]), float(phR[i])])

def make_dichotic_stim(params, fs, rng):
    """
    PLACEHOLDER generator.
    Replace this with your real dichotic pitch generator.
    Must return stereo float32 array shape (nSamples, 2) in [-1, 1].
    """
    dur = float(params["dur"])
    n = int(fs * dur)
    t = np.arange(n) / fs

    noiseL = rng.normal(0, 1, n).astype(np.float32)
    noiseR = rng.normal(0, 1, n).astype(np.float32)

    f = float(params.get("carrier_hz", 500.0))
    phase = float(params.get("phase_offset_rad", 0.0))
    tone = np.sin(2*np.pi*f*t).astype(np.float32)

    if params.get("is_sham", False):
        yL = noiseL
        yR = noiseR
    else:
        yL = noiseL + 0.15 * tone
        yR = noiseR + 0.15 * np.sin(2*np.pi*f*t + phase).astype(np.float32)

    y = np.column_stack([yL, yR]).astype(np.float32)
    y = normalize(y)
    y = np.clip(y, -1.0, 1.0)
    return y

def make_melody(params, fs, rng):
    pitch_seq = params["pitch_seq_hz"]
    note_dur = float(params["note_dur"])
    gap = float(params["gap"])
    chunks = []
    for f in pitch_seq:
        p = {
            "dur": note_dur,
            "carrier_hz": float(f),
            "phase_offset_rad": float(params.get("phase_offset_rad", 0.0)),
            "is_sham": False
        }
        note = make_dichotic_stim(p, fs, rng)
        chunks.append(note)
        if gap > 0:
            chunks.append(np.zeros((int(fs*gap), 2), dtype=np.float32))
    y = np.vstack(chunks)
    y = normalize(y)
    y = np.clip(y, -1.0, 1.0)
    return y

def save_wav_and_fft(stereo, fs, wav_path, fft_csv_path, run_id=None, stim_id=None, stim_type=None, center_freq_hz=None):
    stereo = np.asarray(stereo, dtype=np.float32)
    stereo = normalize(stereo)
    stereo = np.clip(stereo, -1.0, 1.0)
    sf.write(wav_path, stereo, fs, subtype="PCM_16")
    compute_fft_csv(stereo, fs, fft_csv_path, run_id=run_id, stim_id=stim_id, stim_type=stim_type, center_freq_hz=center_freq_hz)

def write_manifest(manifest_path, data):
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
