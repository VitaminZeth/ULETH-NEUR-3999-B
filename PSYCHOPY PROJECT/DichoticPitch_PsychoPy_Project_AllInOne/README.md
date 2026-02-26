# Dichotic Pitch PsychoPy Project (All-in-one)

This project is a **working PsychoPy Coder** experiment that implements the phase flow you described, including:
- Phase A: File acquisition (generates 5 stimuli, 3 shams, 2 melodies; stereo WAV)
- Per-WAV FFT CSV export (Frequency, Mag_L dBFS, Mag_R dBFS, Phase_L, Phase_R)
- Master logs (manifest.json, stim_features.csv, trial_list.csv, melody_list.csv)
- Phase B: Calibration (volume + left/right orientation beeps)
- Phase C: Hearing test (fixed frequencies; simple threshold search per ear)
- Phase D: Preparation (applies user volume + audiogram-derived per-ear gain to make compensated copies)
- Phase E: Trials + Melodic trials (randomized playback; responses logged to both PsychoPy data and responses.csv)

## How to run
1. Open PsychoPy.
2. Go to **Coder**.
3. Open `run_experiment.py` and run.

## Outputs
A per-run folder is created at:
`OUTPUT/RUN_YYYYMMDD_HHMMSS/...`
with subfolders for Phase A audio and FFT exports, and Phase E response logs.

## Notes
- The dichotic pitch generation in `stimuli.py` contains a **placeholder** generator (`make_dichotic_stim`).
  Replace it with your real dichotic pitch code (must return stereo float array Nx2 in [-1,1]).
- This project avoids the Builder `os` scoping error by keeping imports at module top-level.
- If you still want a Builder version, this Coder project is the safest reference;
  you can port each phase into Builder routines using Code Components.
