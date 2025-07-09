import sounddevice as sd
import numpy as np  

samplerate = 44100  # samples per second
duration = 3        # seconds
frequency_left = 440 # Hz (A4)
frequency_right = 660 # Hz (E5)

    # Generate time array
t = np.linspace(0, duration, int(samplerate * duration), endpoint=False)

    # Generate left channel data
left_channel = 0.5 * np.sin(2 * np.pi * frequency_left * t)

    # Generate right channel data
right_channel = 0.5 * np.sin(2 * np.pi * frequency_right * t)

    # Combine into a stereo array (two columns)
stereo_data = np.column_stack((left_channel, right_channel))
    
sd.play(stereo_data, samplerate)
sd.wait() # Wait until playback is finished

# Explanation:
# sounddevice expects multi-channel audio data to be in a NumPy array where each column represents a distinct channel. For stereo, this means the array will have a shape like (num_samples, 2).
# When you pass this two-column array to sd.play(), sounddevice automatically interprets it as stereo data and sends the first column to the left channel and the second column to the right channel of your default output device.
# The samplerate argument is crucial for correct playback speed.
# sd.wait() is used to block the script execution until the audio playback is complete.