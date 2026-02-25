#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2026.1.0),
    on February 25, 2026, at 09:48
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (
    NOT_STARTED, STARTED, PLAYING, PAUSED, STOPPED, STOPPING, FINISHED, PRESSED, 
    RELEASED, FOREVER, priority
)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2026.1.0'
expName = '20260225_DichoticPitchProject'  # from the Builder filename that created this script
expVersion = ''
# a list of functions to run when the experiment ends (starts off blank)
runAtExit = []
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'expVersion|hid': expVersion,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = (1024, 768)
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # replace default participant ID
    if prefs.piloting['replaceParticipantID']:
        expInfo['participant'] = 'pilot'

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version=expVersion,
        extraInfo=expInfo, runtimeInfo=None,
        originPath='D:\\[Fork]\\ULETH-NEUR-3999-B\\PsychoPy Projects\\dichotic_pitch_project_V5\\20260225_DichoticPitchProject_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    # store pilot mode in data file
    thisExp.addData('piloting', PILOTING, priority=priority.LOW)
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    if PILOTING:
        # show a visual indicator if we're in piloting mode
        if prefs.piloting['showPilotingIndicator']:
            win.showPilotingIndicator()
        # always show the mouse in piloting mode
        if prefs.piloting['forceMouseVisible']:
            win.mouseVisible = True
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    ioSession = ioServer = eyetracker = None
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ptb'
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], currentRoutine=None):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    currentRoutine : psychopy.data.Routine
        Current Routine we are in at time of pausing, if any. This object tells PsychoPy what Components to pause/play/dispatch.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='PsychToolbox',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # dispatch messages on response components
        if currentRoutine is not None:
            for comp in currentRoutine.getDispatchComponents():
                comp.device.dispatchMessages()
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # update experiment info
    expInfo['date'] = data.getDateStr()
    expInfo['expName'] = expName
    expInfo['expVersion'] = expVersion
    expInfo['psychopyVersion'] = psychopyVersion
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='PsychToolbox'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "PHASE_A_FileAcquisition" ---
    text = visual.TextStim(win=win, name='text',
        text='Use up/down to set volume. Press space to continue.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    keyList = keyboard.Keyboard(deviceName='defaultKeyboard')
    # Run 'Begin Experiment' code from acq_code
    # --- Phase A (AcquireFiles) : Builder Code Component -> Begin Experiment ---
    # This block sets up labeled folders + per-WAV FFT CSV export + stimulus feature logs.
    # You will use the helper function save_wav_and_fft(...) inside your stimulus-generation loops.
    
    import os, json, time, csv
    import numpy as np
    from psychopy import sound
    import soundfile as sf
    
    # =========================
    # Folder structure per run
    # =========================
    root_out = os.path.join(thisExp.runtimePath, "OUTPUT")
    os.makedirs(root_out, exist_ok=True)
    
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(root_out, f"RUN_{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Phase A folders
    phaseA_dir  = os.path.join(run_dir, "PHASEA_FileAcquisition")
    phaseA_wav  = os.path.join(phaseA_dir, "WAV")
    phaseA_fft  = os.path.join(phaseA_dir, "FFT_CSV")
    phaseA_logs = os.path.join(phaseA_dir, "LOGS")
    
    for p in (phaseA_dir, phaseA_wav, phaseA_fft, phaseA_logs):
        os.makedirs(p, exist_ok=True)
    
    # PsychoPy audio settings
    fs = 48000
    base_dur = 0.75
    
    # Logs / manifests
    manifest_path      = os.path.join(phaseA_logs, "manifest.json")
    trial_list_path    = os.path.join(phaseA_logs, "trial_list.csv")
    mel_list_path      = os.path.join(phaseA_logs, "melody_list.csv")
    stim_features_csv  = os.path.join(phaseA_logs, "stim_features.csv")
    
    # Random generator (random each run unless you set a seed)
    rng = np.random.default_rng()
    
    # In-memory registry for everything generated this run
    items = []       # shams + stimuli
    mel_items = []   # melodies
    
    # =========================
    # Helper functions
    # =========================
    def normalize(x, peak=0.98):
        m = float(np.max(np.abs(x)))
        if m < 1e-12:
            return x
        return (x / m) * peak
    
    def compute_fft_csv(stereo, fs, out_csv_path, center_freq_hz=None, stim_id=None, stim_type=None):
        """
        Export FFT CSV for stereo signal:
        Frequency_Hz, Mag_L_dBFS, Mag_R_dBFS, Phase_L_rad, Phase_R_rad
        """
        stereo = np.asarray(stereo)
        if stereo.ndim != 2 or stereo.shape[1] != 2:
            raise ValueError("compute_fft_csv expects stereo array of shape (nSamples, 2)")
    
        xL = stereo[:, 0].astype(np.float64)
        xR = stereo[:, 1].astype(np.float64)
    
        XL = np.fft.rfft(xL)
        XR = np.fft.rfft(xR)
        freqs = np.fft.rfftfreq(len(xL), d=1.0/fs)
    
        eps = 1e-12
        # Rough dBFS-ish scaling (good for comparisons; not a calibrated SPL)
        magL = 20.0 * np.log10((np.abs(XL) / (len(xL)/2.0)) + eps)
        magR = 20.0 * np.log10((np.abs(XR) / (len(xR)/2.0)) + eps)
    
        phL = np.angle(XL)
        phR = np.angle(XR)
    
        with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            # metadata rows (easy to parse later)
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
    
    def save_wav_and_fft(stereo, stim_id, stim_type, params, center_freq_hz=None, is_melody=False):
        """
        Saves:
          - stereo WAV into PHASEA/WAV
          - FFT CSV into PHASEA/FFT_CSV
        Registers:
          - items / mel_items list entries with paths + metadata
        """
        stereo = np.asarray(stereo, dtype=np.float32)
        if stereo.ndim != 2 or stereo.shape[1] != 2:
            raise ValueError("save_wav_and_fft expects stereo array of shape (nSamples, 2)")
    
        # safety normalize + clip
        stereo = normalize(stereo)
        stereo = np.clip(stereo, -1.0, 1.0)
    
        wav_name = f"{run_id}_{stim_id}.wav"
        wav_path = os.path.join(phaseA_wav, wav_name)
        sf.write(wav_path, stereo, fs, subtype="PCM_16")
    
        fft_name = f"{run_id}_{stim_id}_FFT.csv"
        fft_path = os.path.join(phaseA_fft, fft_name)
        compute_fft_csv(
            stereo, fs, fft_path,
            center_freq_hz=center_freq_hz,
            stim_id=stim_id,
            stim_type=stim_type
        )
    
        entry = {
            "run_id": run_id,
            "stimId": stim_id,
            "stimType": stim_type,         # "stim" / "sham" / "melody"
            "centerFreqHz": center_freq_hz,
            "wavPath": wav_path,
            "fftCsvPath": fft_path,
            "params": params
        }
        if is_melody:
            mel_items.append(entry)
        else:
            items.append(entry)
    
        return entry
    
    def write_phaseA_logs():
        """
        Call this AFTER you finish generating all shams/stims/melodies.
        Writes:
          - manifest.json (full detail)
          - stim_features.csv (easy spreadsheet view)
          - trial_list.csv and melody_list.csv for Builder loops
        """
        # full manifest
        manifest = {"run_id": run_id, "fs": fs, "items": items, "melodies": mel_items}
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
    
        # simple features CSV
        with open(stim_features_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["run_id","stimId","stimType","centerFreqHz","wavPath","fftCsvPath"])
            for it in items + mel_items:
                w.writerow([it["run_id"], it["stimId"], it["stimType"], it["centerFreqHz"], it["wavPath"], it["fftCsvPath"]])
    
        # Builder trial lists (CSV headers must match your loop variable names)
        with open(trial_list_path, "w", encoding="utf-8") as f:
            f.write("stimFile,stimType,stimId,centerFreqHz,fftCsvPath\n")
            for it in items:
                f.write(f"{it['wavPath']},{it['stimType']},{it['stimId']},{it['centerFreqHz']},{it['fftCsvPath']}\n")
    
        with open(mel_list_path, "w", encoding="utf-8") as f:
            f.write("stimFile,stimType,stimId,centerFreqHz,fftCsvPath\n")
            for it in mel_items:
                f.write(f"{it['wavPath']},{it['stimType']},{it['stimId']},{it['centerFreqHz']},{it['fftCsvPath']}\n")
    
        # make paths available to later Builder loops/routines
        expInfo["run_id"] = run_id
        expInfo["phaseA_dir"] = phaseA_dir
        expInfo["trial_list_path"] = trial_list_path
        expInfo["mel_list_path"] = mel_list_path
        expInfo["manifest_path"] = manifest_path
        expInfo["stim_features_csv"] = stim_features_csv
    
    # -------------------------
    # HOW TO USE IN YOUR LOOPS:
    # -------------------------
    # Wherever you currently do:
    #   sf.write(fpath, y, fs, subtype="PCM_16")
    # Replace it with:
    #   save_wav_and_fft(
    #       stereo=y,
    #       stim_id=params["id"],
    #       stim_type=params["kind"],           # "sham" or "stim" or "melody"
    #       params=params,
    #       center_freq_hz=params.get("carrier_hz", None),
    #       is_melody=(params["kind"] == "melody")
    #   )
    #
    # After generating all files, call once:
    #   write_phaseA_logs()
    
    # --- Initialize components for Routine "PHASE_B__Calibration" ---
    # Run 'Begin Experiment' code from cal_code
    # Default volume scalar (0.0 - 1.0). We'll store it and apply later.
    hp_volume = 0.3
    
    # --- Initialize components for Routine "HearingTest" ---
    # Run 'Begin Experiment' code from hear_code
    from collections import defaultdict
    
    # Fixed test freqs (set frequencies per trial, as requested)
    test_freqs = [250, 500, 1000, 2000, 4000, 8000]
    
    # Simple threshold search levels (dB-like steps mapped to linear)
    # You can tune these levels. Start easier -> harder
    levels = [0.25, 0.18, 0.12, 0.08, 0.05, 0.03, 0.02]  # linear amplitude
    # Store results: threshold = lowest level heard (per ear)
    audiogram = {"L": {}, "R": {}}
    yesno = keyboard.Keyboard(deviceName='defaultKeyboard')
    Confirm = visual.TextStim(win=win, name='Confirm',
        text='Did you hear it? (Y/N)',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "PHASE_D_Preperation" ---
    # Run 'Begin Experiment' code from prep_code
    import json
    import numpy as np
    import soundfile as sf
    import os
    
    # Load manifest
    with open(expInfo["manifest_path"], "r", encoding="utf-8") as f:
        manifest = json.load(f)
    
    fs = int(manifest["fs"])
    hp_volume = float(expInfo.get("hp_volume", 0.3))
    
    # Convert audiogram thresholds into ear gains.
    # Simple rule: if threshold is high (needs louder), boost more.
    # If None (not heard at max), boost strongly but cap.
    def ear_gain_map(ear_dict):
        gains = {}
        for freq, thr in ear_dict.items():
            if thr is None:
                gains[freq] = 2.5  # big boost cap
            else:
                # normalize relative to easiest level
                # smaller thr means good hearing -> less boost
                gains[freq] = float(np.clip(levels[0] / max(thr, 1e-6), 0.8, 2.5))
        return gains
    
    gL = ear_gain_map(audiogram["L"])
    gR = ear_gain_map(audiogram["R"])
    
    # For a simple first version: use the median gain across tested freqs as a broadband per-ear gain
    L_gain = float(np.median(list(gL.values())))
    R_gain = float(np.median(list(gR.values())))
    
    comp_dir = os.path.join(out_dir, "compensated_audio")
    os.makedirs(comp_dir, exist_ok=True)
    
    def compensate_file(in_path):
        y, fsi = sf.read(in_path, dtype="float32", always_2d=True)
        if fsi != fs:
            # If mismatch, you should resample; keeping simple here:
            pass
        y[:, 0] *= L_gain
        y[:, 1] *= R_gain
        # normalize + safe cap
        y = np.clip(y, -1.0, 1.0)
        return y
    
    # Write compensated trial lists
    trial_list_comp = os.path.join(comp_dir, "trial_list_COMP.csv")
    mel_list_comp = os.path.join(comp_dir, "melody_list_COMP.csv")
    
    with open(trial_list_comp, "w", encoding="utf-8") as f:
        f.write("stimFile,stimType,stimId\n")
        for it in manifest["items"]:
            in_path = it["file"]
            base = os.path.splitext(os.path.basename(in_path))[0]
            out_path = os.path.join(comp_dir, base + "_COMP.wav")
            y = compensate_file(in_path)
            sf.write(out_path, y, fs, subtype="PCM_16")
            f.write(f"{out_path},{it['type']},{it['id']}\n")
    
    with open(mel_list_comp, "w", encoding="utf-8") as f:
        f.write("stimFile,stimType,stimId\n")
        for it in manifest["melodies"]:
            in_path = it["file"]
            base = os.path.splitext(os.path.basename(in_path))[0]
            out_path = os.path.join(comp_dir, base + "_COMP.wav")
            y = compensate_file(in_path)
            sf.write(out_path, y, fs, subtype="PCM_16")
            f.write(f"{out_path},{it['type']},{it['id']}\n")
    
    # Store paths for trial loops
    expInfo["trial_list_comp"] = trial_list_comp
    expInfo["mel_list_comp"] = mel_list_comp
    
    # Store gains used
    thisExp.addData("audiogram_L_gain", L_gain)
    thisExp.addData("audiogram_R_gain", R_gain)
    
    # --- Initialize components for Routine "Phase_E_Trial_Phase" ---
    # set audio backend
    sound.Sound.backend = 'ptb'
    SoundComponent = sound.Sound(
        'A', 
        secs=1.0, 
        stereo=True, 
        hamming=True, 
        speaker=None,    name='SoundComponent'
    )
    SoundComponent.setVolume(1.0)
    yesno2 = keyboard.Keyboard(deviceName='defaultKeyboard')
    text_2 = visual.TextStim(win=win, name='text_2',
        text='Did you hear the pitch?',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "MelodicTrials" ---
    sound_1 = sound.Sound(
        'A', 
        secs=1.0, 
        stereo=True, 
        hamming=True, 
        speaker=None,    name='sound_1'
    )
    sound_1.setVolume(1.0)
    text_3 = visual.TextStim(win=win, name='text_3',
        text='How many pitches did you hear? (0–6)',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    melodic_resp = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    if eyetracker is not None:
        eyetracker.enableEventReporting()
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "PHASE_A_FileAcquisition" ---
    # create an object to store info about Routine PHASE_A_FileAcquisition
    PHASE_A_FileAcquisition = data.Routine(
        name='PHASE_A_FileAcquisition',
        components=[text, keyList],
    )
    PHASE_A_FileAcquisition.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for keyList
    keyList.keys = []
    keyList.rt = []
    _keyList_allKeys = []
    # Run 'Begin Routine' code from acq_code
    def normalize(x, peak=0.98):
        m = np.max(np.abs(x))
        if m < 1e-9:
            return x
        return (x / m) * peak
    
    def make_dichotic_stim(params, fs):
        """
        Replace this with YOUR dichotic pitch generator.
        Must return stereo float32 array shape (nSamples, 2) in [-1, 1].
        """
        dur = params["dur"]
        n = int(fs * dur)
        t = np.arange(n) / fs
    
        # Example placeholder: noise with a subtle phase manipulation
        noiseL = rng.normal(0, 1, n)
        noiseR = rng.normal(0, 1, n)
    
        # Fake "illusory pitch band" control param
        f = params["carrier_hz"]
        tone = np.sin(2*np.pi*f*t)
    
        if params["is_sham"]:
            # sham: no added structure
            yL = noiseL
            yR = noiseR
        else:
            # “stim”: add correlated structure differently per ear
            yL = noiseL + 0.15 * tone
            yR = noiseR + 0.15 * np.sin(2*np.pi*f*t + params["phase_offset_rad"])
    
        y = np.column_stack([yL, yR]).astype(np.float32)
        y = normalize(y)
        return y
    
    def make_melody(params, fs):
        """
        6 pitches in sequence. Returns stereo float32 array.
        """
        pitch_seq = params["pitch_seq_hz"]  # length 6
        note_dur = params["note_dur"]
        gap = params["gap"]
        chunks = []
        for f in pitch_seq:
            n = int(fs * note_dur)
            t = np.arange(n) / fs
            # Use the same dichotic “style” but change pitch each note:
            p = {
                "dur": note_dur,
                "carrier_hz": float(f),
                "phase_offset_rad": params["phase_offset_rad"],
                "is_sham": False
            }
            note = make_dichotic_stim(p, fs)
            chunks.append(note)
            if gap > 0:
                chunks.append(np.zeros((int(fs*gap), 2), dtype=np.float32))
        y = np.vstack(chunks)
        y = normalize(y)
        return y
    
    # --- Generate 3 shams + 5 stimuli ---
    items = []
    ts = time.strftime("%Y%m%d_%H%M%S")
    
    # 3 shams
    for i in range(3):
        params = {
            "kind": "sham",
            "dur": base_dur,
            "carrier_hz": float(rng.choice([200, 250, 315, 400, 500, 630])),
            "phase_offset_rad": 0.0,
            "is_sham": True,
            "id": f"sham_{i+1}"
        }
        y = make_dichotic_stim(params, fs)
        fname = f"{ts}_{params['id']}.wav"
        fpath = os.path.join(out_dir, fname)
        sf.write(fpath, y, fs, subtype="PCM_16")
        items.append({"file": fpath, "type": "sham", "id": params["id"], "params": params})
    
    # 5 stimuli
    for i in range(5):
        params = {
            "kind": "stim",
            "dur": base_dur,
            "carrier_hz": float(rng.choice([200, 250, 315, 400, 500, 630])),
            "phase_offset_rad": float(rng.uniform(-np.pi, np.pi)),
            "is_sham": False,
            "id": f"stim_{i+1}"
        }
        y = make_dichotic_stim(params, fs)
        fname = f"{ts}_{params['id']}.wav"
        fpath = os.path.join(out_dir, fname)
        sf.write(fpath, y, fs, subtype="PCM_16")
        items.append({"file": fpath, "type": "stim", "id": params["id"], "params": params})
    
    # --- Generate 2 melodic files (each 6 pitches) ---
    mel_items = []
    for i in range(2):
        pitch_pool = np.array([220, 247, 277, 311, 349, 392, 440, 494], dtype=float)
        pitch_seq = rng.choice(pitch_pool, size=6, replace=True).tolist()
        params = {
            "kind": "melody",
            "pitch_seq_hz": pitch_seq,
            "note_dur": 0.35,
            "gap": 0.05,
            "phase_offset_rad": float(rng.uniform(-np.pi, np.pi)),
            "id": f"melody_{i+1}"
        }
        y = make_melody(params, fs)
        fname = f"{ts}_{params['id']}.wav"
        fpath = os.path.join(out_dir, fname)
        sf.write(fpath, y, fs, subtype="PCM_16")
        mel_items.append({"file": fpath, "type": "melody", "id": params["id"], "params": params})
    
    # Save manifest (for regeneration/compensation)
    manifest = {"fs": fs, "items": items, "melodies": mel_items, "timestamp": ts}
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    
    # Write CSV trial lists Builder can read (simple columns)
    with open(trial_list_path, "w", encoding="utf-8") as f:
        f.write("stimFile,stimType,stimId\n")
        for it in items:
            f.write(f"{it['file']},{it['type']},{it['id']}\n")
    
    with open(mel_list_path, "w", encoding="utf-8") as f:
        f.write("stimFile,stimType,stimId\n")
        for it in mel_items:
            f.write(f"{it['file']},{it['type']},{it['id']}\n")
    
    # Make paths available later
    expInfo["trial_list_path"] = trial_list_path
    expInfo["mel_list_path"] = mel_list_path
    expInfo["manifest_path"] = manifest_path
    # store start times for PHASE_A_FileAcquisition
    PHASE_A_FileAcquisition.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    PHASE_A_FileAcquisition.tStart = globalClock.getTime(format='float')
    PHASE_A_FileAcquisition.status = STARTED
    thisExp.addData('PHASE_A_FileAcquisition.started', PHASE_A_FileAcquisition.tStart)
    PHASE_A_FileAcquisition.maxDuration = None
    # keep track of which components have finished
    PHASE_A_FileAcquisitionComponents = PHASE_A_FileAcquisition.components
    for thisComponent in PHASE_A_FileAcquisition.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "PHASE_A_FileAcquisition" ---
    thisExp.currentRoutine = PHASE_A_FileAcquisition
    PHASE_A_FileAcquisition.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text* updates
        
        # if text is starting this frame...
        if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text.frameNStart = frameN  # exact frame index
            text.tStart = t  # local t and not account for scr refresh
            text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text.started')
            # update status
            text.status = STARTED
            text.setAutoDraw(True)
        
        # if text is active this frame...
        if text.status == STARTED:
            # update params
            pass
        
        # if text is stopping this frame...
        if text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                text.tStop = t  # not accounting for scr refresh
                text.tStopRefresh = tThisFlipGlobal  # on global time
                text.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text.stopped')
                # update status
                text.status = FINISHED
                text.setAutoDraw(False)
        
        # *keyList* updates
        waitOnFlip = False
        
        # if keyList is starting this frame...
        if keyList.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            keyList.frameNStart = frameN  # exact frame index
            keyList.tStart = t  # local t and not account for scr refresh
            keyList.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(keyList, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'keyList.started')
            # update status
            keyList.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(keyList.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(keyList.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if keyList.status == STARTED and not waitOnFlip:
            theseKeys = keyList.getKeys(keyList=['up','down','left','right','space','escape'], ignoreKeys=["escape"], waitRelease=False)
            _keyList_allKeys.extend(theseKeys)
            if len(_keyList_allKeys):
                keyList.keys = _keyList_allKeys[-1].name  # just the last key pressed
                keyList.rt = _keyList_allKeys[-1].rt
                keyList.duration = _keyList_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=PHASE_A_FileAcquisition,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            PHASE_A_FileAcquisition.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if PHASE_A_FileAcquisition.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in PHASE_A_FileAcquisition.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "PHASE_A_FileAcquisition" ---
    for thisComponent in PHASE_A_FileAcquisition.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for PHASE_A_FileAcquisition
    PHASE_A_FileAcquisition.tStop = globalClock.getTime(format='float')
    PHASE_A_FileAcquisition.tStopRefresh = tThisFlipGlobal
    thisExp.addData('PHASE_A_FileAcquisition.stopped', PHASE_A_FileAcquisition.tStop)
    # check responses
    if keyList.keys in ['', [], None]:  # No response was made
        keyList.keys = None
    thisExp.addData('keyList.keys',keyList.keys)
    if keyList.keys != None:  # we had a response
        thisExp.addData('keyList.rt', keyList.rt)
        thisExp.addData('keyList.duration', keyList.duration)
    thisExp.nextEntry()
    # the Routine "PHASE_A_FileAcquisition" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "PHASE_B__Calibration" ---
    # create an object to store info about Routine PHASE_B__Calibration
    PHASE_B__Calibration = data.Routine(
        name='PHASE_B__Calibration',
        components=[],
    )
    PHASE_B__Calibration.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from cal_code
    # Simple stereo orientation test sounds
    fs_cal = 48000
    t = np.arange(int(fs_cal*0.25))/fs_cal
    beep = (0.2*np.sin(2*np.pi*800*t)).astype(np.float32)
    
    left_beep = np.column_stack([beep, np.zeros_like(beep)])
    right_beep = np.column_stack([np.zeros_like(beep), beep])
    
    snd_left = sound.Sound(value=left_beep, sampleRate=fs_cal, stereo=True)
    snd_right = sound.Sound(value=right_beep, sampleRate=fs_cal, stereo=True)
    
    # apply current volume
    snd_left.setVolume(hp_volume)
    snd_right.setVolume(hp_volume)
    
    cal_stage = 0  # 0 = waiting, 1 = played left, 2 = played right
    # store start times for PHASE_B__Calibration
    PHASE_B__Calibration.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    PHASE_B__Calibration.tStart = globalClock.getTime(format='float')
    PHASE_B__Calibration.status = STARTED
    thisExp.addData('PHASE_B__Calibration.started', PHASE_B__Calibration.tStart)
    PHASE_B__Calibration.maxDuration = None
    # keep track of which components have finished
    PHASE_B__CalibrationComponents = PHASE_B__Calibration.components
    for thisComponent in PHASE_B__Calibration.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "PHASE_B__Calibration" ---
    thisExp.currentRoutine = PHASE_B__Calibration
    PHASE_B__Calibration.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # Run 'Each Frame' code from cal_code
        keys = cal_kb.getKeys(keyList=['up','down','left','right','space'], waitRelease=False)
        for k in keys:
            if k.name == 'up':
                hp_volume = min(1.0, hp_volume + 0.02)
            elif k.name == 'down':
                hp_volume = max(0.0, hp_volume - 0.02)
            elif k.name == 'left':
                snd_left.setVolume(hp_volume)
                snd_left.play()
            elif k.name == 'right':
                snd_right.setVolume(hp_volume)
                snd_right.play()
            elif k.name == 'space':
                continueRoutine = False
        
        # (Optional) update a Text component like: f"Volume: {hp_volume:.2f}"
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=PHASE_B__Calibration,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            PHASE_B__Calibration.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if PHASE_B__Calibration.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in PHASE_B__Calibration.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "PHASE_B__Calibration" ---
    for thisComponent in PHASE_B__Calibration.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for PHASE_B__Calibration
    PHASE_B__Calibration.tStop = globalClock.getTime(format='float')
    PHASE_B__Calibration.tStopRefresh = tThisFlipGlobal
    thisExp.addData('PHASE_B__Calibration.stopped', PHASE_B__Calibration.tStop)
    # Run 'End Routine' code from cal_code
    # store chosen volume
    expInfo["hp_volume"] = hp_volume
    thisExp.addData("hp_volume", hp_volume)
    thisExp.nextEntry()
    # the Routine "PHASE_B__Calibration" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    hearing_loop = data.TrialHandler2(
        name='hearing_loop',
        nReps=2*len(test_freqs), 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(hearing_loop)  # add the loop to the experiment
    thisHearing_loop = hearing_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisHearing_loop.rgb)
    if thisHearing_loop != None:
        for paramName in thisHearing_loop:
            globals()[paramName] = thisHearing_loop[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisHearing_loop in hearing_loop:
        hearing_loop.status = STARTED
        if hasattr(thisHearing_loop, 'status'):
            thisHearing_loop.status = STARTED
        currentLoop = hearing_loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisHearing_loop.rgb)
        if thisHearing_loop != None:
            for paramName in thisHearing_loop:
                globals()[paramName] = thisHearing_loop[paramName]
        
        # --- Prepare to start Routine "HearingTest" ---
        # create an object to store info about Routine HearingTest
        HearingTest = data.Routine(
            name='HearingTest',
            components=[yesno, Confirm],
        )
        HearingTest.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from hear_code
        # Determine which frequency and ear this iteration is (sequential)
        # We run 2 ears x len(test_freqs). If you want multiple repeats, set nReps>1.
        trial_index = hearing_loop.thisN  # 0..(2*len(freqs)-1)
        ear = "L" if trial_index < len(test_freqs) else "R"
        freq = test_freqs[trial_index % len(test_freqs)]
        
        # Simple descending method: play from loud to soft until "no"
        level_index = 0
        heard_last = True
        threshold_found = False
        
        # Build current tone
        fs_ht = 48000
        dur_ht = 0.5
        t = np.arange(int(fs_ht*dur_ht))/fs_ht
        tone = np.sin(2*np.pi*freq*t).astype(np.float32)
        
        def make_ear_tone(amp, ear):
            x = (amp * tone).astype(np.float32)
            if ear == "L":
                return np.column_stack([x, np.zeros_like(x)])
            else:
                return np.column_stack([np.zeros_like(x), x])
        
        current_amp = levels[level_index]
        stim_arr = make_ear_tone(current_amp, ear)
        ht_sound = sound.Sound(value=stim_arr, sampleRate=fs_ht, stereo=True)
        ht_sound.setVolume(float(expInfo.get("hp_volume", 0.3)))  # user calibration volume
        played_once = False
        # create starting attributes for yesno
        yesno.keys = []
        yesno.rt = []
        _yesno_allKeys = []
        # store start times for HearingTest
        HearingTest.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        HearingTest.tStart = globalClock.getTime(format='float')
        HearingTest.status = STARTED
        thisExp.addData('HearingTest.started', HearingTest.tStart)
        HearingTest.maxDuration = None
        # keep track of which components have finished
        HearingTestComponents = HearingTest.components
        for thisComponent in HearingTest.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "HearingTest" ---
        thisExp.currentRoutine = HearingTest
        HearingTest.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisHearing_loop, 'status') and thisHearing_loop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from hear_code
            # Play once at start
            if not played_once:
                ht_sound.play()
                played_once = True
            
            keys = ht_kb.getKeys(keyList=['y','n'], waitRelease=False)
            if keys:
                resp = keys[-1].name
            
                if resp == 'y':
                    # go softer
                    if level_index < len(levels) - 1:
                        level_index += 1
                        current_amp = levels[level_index]
                        stim_arr = make_ear_tone(current_amp, ear)
                        ht_sound = sound.Sound(value=stim_arr, sampleRate=fs_ht, stereo=True)
                        ht_sound.setVolume(float(expInfo.get("hp_volume", 0.3)))
                        played_once = False
                    else:
                        # heard even at softest
                        audiogram[ear][freq] = levels[-1]
                        threshold_found = True
                        continueRoutine = False
            
                elif resp == 'n':
                    # threshold is previous level (if any), else none
                    if level_index == 0:
                        audiogram[ear][freq] = None  # not heard at max level
                    else:
                        audiogram[ear][freq] = levels[level_index - 1]
                    threshold_found = True
                    continueRoutine = False
            
            # *yesno* updates
            waitOnFlip = False
            
            # if yesno is starting this frame...
            if yesno.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                yesno.frameNStart = frameN  # exact frame index
                yesno.tStart = t  # local t and not account for scr refresh
                yesno.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(yesno, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'yesno.started')
                # update status
                yesno.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(yesno.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(yesno.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if yesno.status == STARTED and not waitOnFlip:
                theseKeys = yesno.getKeys(keyList=['y','n'], ignoreKeys=["escape"], waitRelease=False)
                _yesno_allKeys.extend(theseKeys)
                if len(_yesno_allKeys):
                    yesno.keys = _yesno_allKeys[-1].name  # just the last key pressed
                    yesno.rt = _yesno_allKeys[-1].rt
                    yesno.duration = _yesno_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *Confirm* updates
            
            # if Confirm is starting this frame...
            if Confirm.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Confirm.frameNStart = frameN  # exact frame index
                Confirm.tStart = t  # local t and not account for scr refresh
                Confirm.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Confirm, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Confirm.started')
                # update status
                Confirm.status = STARTED
                Confirm.setAutoDraw(True)
            
            # if Confirm is active this frame...
            if Confirm.status == STARTED:
                # update params
                pass
            
            # if Confirm is stopping this frame...
            if Confirm.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Confirm.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    Confirm.tStop = t  # not accounting for scr refresh
                    Confirm.tStopRefresh = tThisFlipGlobal  # on global time
                    Confirm.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Confirm.stopped')
                    # update status
                    Confirm.status = FINISHED
                    Confirm.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=HearingTest,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                HearingTest.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if HearingTest.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in HearingTest.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "HearingTest" ---
        for thisComponent in HearingTest.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for HearingTest
        HearingTest.tStop = globalClock.getTime(format='float')
        HearingTest.tStopRefresh = tThisFlipGlobal
        thisExp.addData('HearingTest.stopped', HearingTest.tStop)
        # Run 'End Routine' code from hear_code
        thisExp.addData("ht_ear", ear)
        thisExp.addData("ht_freq", freq)
        thisExp.addData("ht_threshold_amp", audiogram[ear][freq])
        # check responses
        if yesno.keys in ['', [], None]:  # No response was made
            yesno.keys = None
        hearing_loop.addData('yesno.keys',yesno.keys)
        if yesno.keys != None:  # we had a response
            hearing_loop.addData('yesno.rt', yesno.rt)
            hearing_loop.addData('yesno.duration', yesno.duration)
        # the Routine "HearingTest" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisHearing_loop as finished
        if hasattr(thisHearing_loop, 'status'):
            thisHearing_loop.status = FINISHED
        # if awaiting a pause, pause now
        if hearing_loop.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            hearing_loop.status = STARTED
        thisExp.nextEntry()
        
    # completed 2*len(test_freqs) repeats of 'hearing_loop'
    hearing_loop.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "PHASE_D_Preperation" ---
    # create an object to store info about Routine PHASE_D_Preperation
    PHASE_D_Preperation = data.Routine(
        name='PHASE_D_Preperation',
        components=[],
    )
    PHASE_D_Preperation.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for PHASE_D_Preperation
    PHASE_D_Preperation.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    PHASE_D_Preperation.tStart = globalClock.getTime(format='float')
    PHASE_D_Preperation.status = STARTED
    thisExp.addData('PHASE_D_Preperation.started', PHASE_D_Preperation.tStart)
    PHASE_D_Preperation.maxDuration = None
    # keep track of which components have finished
    PHASE_D_PreperationComponents = PHASE_D_Preperation.components
    for thisComponent in PHASE_D_Preperation.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "PHASE_D_Preperation" ---
    thisExp.currentRoutine = PHASE_D_Preperation
    PHASE_D_Preperation.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=PHASE_D_Preperation,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            PHASE_D_Preperation.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if PHASE_D_Preperation.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in PHASE_D_Preperation.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "PHASE_D_Preperation" ---
    for thisComponent in PHASE_D_Preperation.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for PHASE_D_Preperation
    PHASE_D_Preperation.tStop = globalClock.getTime(format='float')
    PHASE_D_Preperation.tStopRefresh = tThisFlipGlobal
    thisExp.addData('PHASE_D_Preperation.stopped', PHASE_D_Preperation.tStop)
    thisExp.nextEntry()
    # the Routine "PHASE_D_Preperation" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trial_loop = data.TrialHandler2(
        name='trial_loop',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(trial_loop)  # add the loop to the experiment
    thisTrial_loop = trial_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial_loop.rgb)
    if thisTrial_loop != None:
        for paramName in thisTrial_loop:
            globals()[paramName] = thisTrial_loop[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrial_loop in trial_loop:
        trial_loop.status = STARTED
        if hasattr(thisTrial_loop, 'status'):
            thisTrial_loop.status = STARTED
        currentLoop = trial_loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_loop.rgb)
        if thisTrial_loop != None:
            for paramName in thisTrial_loop:
                globals()[paramName] = thisTrial_loop[paramName]
        
        # --- Prepare to start Routine "Phase_E_Trial_Phase" ---
        # create an object to store info about Routine Phase_E_Trial_Phase
        Phase_E_Trial_Phase = data.Routine(
            name='Phase_E_Trial_Phase',
            components=[SoundComponent, yesno2, text_2],
        )
        Phase_E_Trial_Phase.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        SoundComponent.setSound(stimFile, secs=1.0, hamming=True)
        SoundComponent.setVolume(1.0, log=False)
        SoundComponent.seek(0)
        # create starting attributes for yesno2
        yesno2.keys = []
        yesno2.rt = []
        _yesno2_allKeys = []
        # store start times for Phase_E_Trial_Phase
        Phase_E_Trial_Phase.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Phase_E_Trial_Phase.tStart = globalClock.getTime(format='float')
        Phase_E_Trial_Phase.status = STARTED
        thisExp.addData('Phase_E_Trial_Phase.started', Phase_E_Trial_Phase.tStart)
        Phase_E_Trial_Phase.maxDuration = None
        # keep track of which components have finished
        Phase_E_Trial_PhaseComponents = Phase_E_Trial_Phase.components
        for thisComponent in Phase_E_Trial_Phase.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Phase_E_Trial_Phase" ---
        thisExp.currentRoutine = Phase_E_Trial_Phase
        Phase_E_Trial_Phase.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisTrial_loop, 'status') and thisTrial_loop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *SoundComponent* updates
            
            # if SoundComponent is starting this frame...
            if SoundComponent.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                SoundComponent.frameNStart = frameN  # exact frame index
                SoundComponent.tStart = t  # local t and not account for scr refresh
                SoundComponent.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('SoundComponent.started', tThisFlipGlobal)
                # update status
                SoundComponent.status = STARTED
                SoundComponent.play(when=win)  # sync with win flip
            
            # if SoundComponent is stopping this frame...
            if SoundComponent.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > SoundComponent.tStartRefresh + 1.0-frameTolerance or SoundComponent.isFinished:
                    # keep track of stop time/frame for later
                    SoundComponent.tStop = t  # not accounting for scr refresh
                    SoundComponent.tStopRefresh = tThisFlipGlobal  # on global time
                    SoundComponent.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'SoundComponent.stopped')
                    # update status
                    SoundComponent.status = FINISHED
                    SoundComponent.stop()
            
            # *yesno2* updates
            waitOnFlip = False
            
            # if yesno2 is starting this frame...
            if yesno2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                yesno2.frameNStart = frameN  # exact frame index
                yesno2.tStart = t  # local t and not account for scr refresh
                yesno2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(yesno2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'yesno2.started')
                # update status
                yesno2.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(yesno2.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(yesno2.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if yesno2.status == STARTED and not waitOnFlip:
                theseKeys = yesno2.getKeys(keyList=['y','n'], ignoreKeys=["escape"], waitRelease=False)
                _yesno2_allKeys.extend(theseKeys)
                if len(_yesno2_allKeys):
                    yesno2.keys = _yesno2_allKeys[-1].name  # just the last key pressed
                    yesno2.rt = _yesno2_allKeys[-1].rt
                    yesno2.duration = _yesno2_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *text_2* updates
            
            # if text_2 is starting this frame...
            if text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_2.frameNStart = frameN  # exact frame index
                text_2.tStart = t  # local t and not account for scr refresh
                text_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_2.started')
                # update status
                text_2.status = STARTED
                text_2.setAutoDraw(True)
            
            # if text_2 is active this frame...
            if text_2.status == STARTED:
                # update params
                pass
            
            # if text_2 is stopping this frame...
            if text_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_2.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    text_2.tStop = t  # not accounting for scr refresh
                    text_2.tStopRefresh = tThisFlipGlobal  # on global time
                    text_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_2.stopped')
                    # update status
                    text_2.status = FINISHED
                    text_2.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=Phase_E_Trial_Phase,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                Phase_E_Trial_Phase.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if Phase_E_Trial_Phase.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in Phase_E_Trial_Phase.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Phase_E_Trial_Phase" ---
        for thisComponent in Phase_E_Trial_Phase.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Phase_E_Trial_Phase
        Phase_E_Trial_Phase.tStop = globalClock.getTime(format='float')
        Phase_E_Trial_Phase.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Phase_E_Trial_Phase.stopped', Phase_E_Trial_Phase.tStop)
        SoundComponent.pause()  # ensure sound has stopped at end of Routine
        # check responses
        if yesno2.keys in ['', [], None]:  # No response was made
            yesno2.keys = None
        trial_loop.addData('yesno2.keys',yesno2.keys)
        if yesno2.keys != None:  # we had a response
            trial_loop.addData('yesno2.rt', yesno2.rt)
            trial_loop.addData('yesno2.duration', yesno2.duration)
        # Run 'End Routine' code from code
        thisExp.addData("stimType", stimType)
        thisExp.addData("stimId", stimId)
        thisExp.addData("stimFile", stimFile)
        # the Routine "Phase_E_Trial_Phase" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisTrial_loop as finished
        if hasattr(thisTrial_loop, 'status'):
            thisTrial_loop.status = FINISHED
        # if awaiting a pause, pause now
        if trial_loop.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            trial_loop.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'trial_loop'
    trial_loop.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # set up handler to look after randomisation of conditions etc
    mel_loop = data.TrialHandler2(
        name='mel_loop',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(mel_loop)  # add the loop to the experiment
    thisMel_loop = mel_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisMel_loop.rgb)
    if thisMel_loop != None:
        for paramName in thisMel_loop:
            globals()[paramName] = thisMel_loop[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisMel_loop in mel_loop:
        mel_loop.status = STARTED
        if hasattr(thisMel_loop, 'status'):
            thisMel_loop.status = STARTED
        currentLoop = mel_loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisMel_loop.rgb)
        if thisMel_loop != None:
            for paramName in thisMel_loop:
                globals()[paramName] = thisMel_loop[paramName]
        
        # --- Prepare to start Routine "MelodicTrials" ---
        # create an object to store info about Routine MelodicTrials
        MelodicTrials = data.Routine(
            name='MelodicTrials',
            components=[sound_1, text_3, melodic_resp],
        )
        MelodicTrials.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        sound_1.setSound(stimFile, secs=1.0, hamming=True)
        sound_1.setVolume(1.0, log=False)
        sound_1.seek(0)
        # create starting attributes for melodic_resp
        melodic_resp.keys = []
        melodic_resp.rt = []
        _melodic_resp_allKeys = []
        # store start times for MelodicTrials
        MelodicTrials.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        MelodicTrials.tStart = globalClock.getTime(format='float')
        MelodicTrials.status = STARTED
        thisExp.addData('MelodicTrials.started', MelodicTrials.tStart)
        MelodicTrials.maxDuration = None
        # keep track of which components have finished
        MelodicTrialsComponents = MelodicTrials.components
        for thisComponent in MelodicTrials.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "MelodicTrials" ---
        thisExp.currentRoutine = MelodicTrials
        MelodicTrials.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisMel_loop, 'status') and thisMel_loop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *sound_1* updates
            
            # if sound_1 is starting this frame...
            if sound_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                sound_1.frameNStart = frameN  # exact frame index
                sound_1.tStart = t  # local t and not account for scr refresh
                sound_1.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('sound_1.started', tThisFlipGlobal)
                # update status
                sound_1.status = STARTED
                sound_1.play(when=win)  # sync with win flip
            
            # if sound_1 is stopping this frame...
            if sound_1.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > sound_1.tStartRefresh + 1.0-frameTolerance or sound_1.isFinished:
                    # keep track of stop time/frame for later
                    sound_1.tStop = t  # not accounting for scr refresh
                    sound_1.tStopRefresh = tThisFlipGlobal  # on global time
                    sound_1.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'sound_1.stopped')
                    # update status
                    sound_1.status = FINISHED
                    sound_1.stop()
            
            # *text_3* updates
            
            # if text_3 is starting this frame...
            if text_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_3.frameNStart = frameN  # exact frame index
                text_3.tStart = t  # local t and not account for scr refresh
                text_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_3.started')
                # update status
                text_3.status = STARTED
                text_3.setAutoDraw(True)
            
            # if text_3 is active this frame...
            if text_3.status == STARTED:
                # update params
                pass
            
            # if text_3 is stopping this frame...
            if text_3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_3.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    text_3.tStop = t  # not accounting for scr refresh
                    text_3.tStopRefresh = tThisFlipGlobal  # on global time
                    text_3.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_3.stopped')
                    # update status
                    text_3.status = FINISHED
                    text_3.setAutoDraw(False)
            
            # *melodic_resp* updates
            waitOnFlip = False
            
            # if melodic_resp is starting this frame...
            if melodic_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                melodic_resp.frameNStart = frameN  # exact frame index
                melodic_resp.tStart = t  # local t and not account for scr refresh
                melodic_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(melodic_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'melodic_resp.started')
                # update status
                melodic_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(melodic_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(melodic_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if melodic_resp.status == STARTED and not waitOnFlip:
                theseKeys = melodic_resp.getKeys(keyList=['0','1','2','3','4','5','6'], ignoreKeys=["escape"], waitRelease=False)
                _melodic_resp_allKeys.extend(theseKeys)
                if len(_melodic_resp_allKeys):
                    melodic_resp.keys = _melodic_resp_allKeys[-1].name  # just the last key pressed
                    melodic_resp.rt = _melodic_resp_allKeys[-1].rt
                    melodic_resp.duration = _melodic_resp_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=MelodicTrials,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                MelodicTrials.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if MelodicTrials.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in MelodicTrials.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "MelodicTrials" ---
        for thisComponent in MelodicTrials.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for MelodicTrials
        MelodicTrials.tStop = globalClock.getTime(format='float')
        MelodicTrials.tStopRefresh = tThisFlipGlobal
        thisExp.addData('MelodicTrials.stopped', MelodicTrials.tStop)
        sound_1.pause()  # ensure sound has stopped at end of Routine
        # check responses
        if melodic_resp.keys in ['', [], None]:  # No response was made
            melodic_resp.keys = None
        mel_loop.addData('melodic_resp.keys',melodic_resp.keys)
        if melodic_resp.keys != None:  # we had a response
            mel_loop.addData('melodic_resp.rt', melodic_resp.rt)
            mel_loop.addData('melodic_resp.duration', melodic_resp.duration)
        # Run 'End Routine' code from code_2
        thisExp.addData("melody_id", stimId)
        thisExp.addData("melody_file", stimFile)
        # Keyboard component already logs the key; convert to int if you want:
        # thisExp.addData("melody_count", int(mel_kb.keys[0]) if mel_kb.keys else None)
        # the Routine "MelodicTrials" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisMel_loop as finished
        if hasattr(thisMel_loop, 'status'):
            thisMel_loop.status = FINISHED
        # if awaiting a pause, pause now
        if mel_loop.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            mel_loop.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'mel_loop'
    mel_loop.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    # stop any playback components
    if thisExp.currentRoutine is not None:
        for comp in thisExp.currentRoutine.getPlaybackComponents():
            comp.stop()
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # run any 'at exit' functions
    for fcn in runAtExit:
        fcn()
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
