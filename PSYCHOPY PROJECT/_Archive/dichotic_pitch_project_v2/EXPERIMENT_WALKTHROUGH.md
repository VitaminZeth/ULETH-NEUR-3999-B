
# 🎛️ Experiment Walkthrough: Dichotic Pitch EEG

This document explains how the experiment is structured and what each module does.

---

## 🧪 STAGES

1. **Headphone Check**
   - Two sounds are played: one only in the left ear, one only in the right.
   - The participant identifies which ear hears each tone.

2. **Volume Calibration**
   - A 500Hz tone is played.
   - Participant confirms if it's clearly audible. If not, adjust volume manually.

3. **Practice Trials**
   - A wider-bandwidth version of the illusion is played.
   - Helps familiarize participants with the illusion.

4. **Main Block Structure**
   - Each block includes:
     1. **Control (500 Hz illusion)**
     2. **Three Random Trials** (e.g., 400 Hz phase-flipped)
     3. **Sham Trial** (No phase shift)
     4. **Three More Random Trials**
     5. **Final Control Trial**
     6. **Silent Trial** for EEG noise floor

5. **Breaks**
   - Mid-block rest screen shown.
   - Participant presses spacebar to continue.

---

## 📈 EEG Timing Structure

Each trial:
- Sends a NetStation EEG trigger
- Waits 500ms baseline period
- Plays the audio
- Captures response
- Captures confidence

---

## 🧠 Trial Types

| Type     | Description                      |
|----------|----------------------------------|
| Control  | 500Hz illusion with phase flip   |
| Test     | Random freq (250–1000 Hz) flipped |
| Sham     | No phase shift applied           |
| Practice | Wider bandwidth (e.g., 60Hz)     |
| Silent   | No sound at all (for EEG noise)  |

---

## 📁 File Output Summary

- `/stimuli/*.wav`: Audio files per randomized trial
- `/stimuli/*.csv`: FFT analysis of each sound
- `/data/*.csv`: Trial-level responses, timestamps, events

---

