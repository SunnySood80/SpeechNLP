import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import IPython.display as ipd
from scipy.io import wavfile
from keras.models import load_model
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

st.write('Try these Words: yes, no, up, down, left, right, on, off, stop, go')

model=load_model('SpeechRecogModel.h5')
classes = ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes']

# Define the function that predicts text for the given audio:

def predict(audio):
    prob=model.predict(audio.reshape(1,8000,1))
    index=np.argmax(prob[0])
    return classes[index]


# **The best part is yet to come! Here is a script that prompts a user to record voice commands. Record your own voice commands and test it on the model:**

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_ctx = webrtc_streamer(
    key="WYH",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    #video_processor_factory=VideoProcessor,
    async_processing=True,
)

if st.button(f"Click to Record"):
    '''record_state = st.text("Recording...")
    samplerate = 16000  
    duration = 1  # seconds
    filename = 'yes.wav'
    st.write("Speak Now")
    mydata = sd.rec(int(samplerate * duration), samplerate=samplerate,channels=1, blocking=True)
    st.write("Recording Complete")
    sd.wait()
    sf.write(filename, mydata, samplerate)
    # Let us now read the saved voice command and convert it to text:\
    #reading the voice commands

    samples, sample_rate = librosa.load('yes.wav', sr = 16000)
    samples = librosa.resample(samples, sample_rate, 8000)
    ipd.Audio(samples,rate=8000)      

    #converting voice commands to text

    st.write(predict(samples))'''
    #st.audio(read_audio(filename))
    samples, sample_rate = librosa.load('yes.wav', sr = 16000)
    fig = plt.figure(figsize=(14, 8))
    ax1 = fig.add_subplot(211)
    ax1.set_title('Spectrogram')
    ax1.set_xlabel('time')
    ax1.set_ylabel('Amplitude')
    ax1.plot(np.linspace(0, sample_rate/len(samples), sample_rate), samples)
    st.pyplot(fig)
    audio_file = open('yes.wav', 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes)
