import time
import librosa
import soundfile
import os
from groq import Groq
from transformers import pipeline
import itertools
from config_app.config import get_config

config_app = get_config()
# List of Groq API keys
groq_keys = config_app['parameter']['llm_api_keys']

# Iterator to cycle through Groq API keys
key_iterator = itertools.cycle(groq_keys)

# Function to get the next Groq API key
def get_next_key():
    return next(key_iterator)

# Initialize the first Groq client
current_key = get_next_key()
client = Groq(api_key=current_key)

# Initialize the transcriber with the local model
transcriber = pipeline("automatic-speech-recognition", model="vinai/PhoWhisper-small", device="cpu")

def transcribe_with_api(filename):
    global current_key, client
    retries = len(groq_keys)
    for _ in range(retries):
        try:
            with open(filename, "rb") as file:
                transcription = client.audio.transcriptions.create(
                    file=(filename, file.read()),
                    model="whisper-large-v3",
                    prompt="Specify context or spelling",  # Optional
                    response_format="json",  # Optional
                    language="vi",  # Optional
                    temperature=0.0  # Optional
                )
            return transcription.text
        except Exception as e:
            print(f"Error with key {current_key}: {e}")
            current_key = get_next_key()
            client = Groq(api_key=current_key)
            print(f"Switched to key {current_key}")
            time.sleep(1)  # Add delay if necessary to handle rate limits
    # If all keys fail, raise an exception
    raise Exception("All API keys failed")

def transcribe_with_local_model(path_audio):
    t0 = time.time()
    text = transcriber(path_audio)["text"]
    print('speech_2_text time: ', time.time() - t0)
    return text

def downsampleWav(sound, dst):
    y, s = librosa.load(sound, sr=16000)
    soundfile.write(dst, y, s)

def speech_2_text(path_audio: bytes):
    try:
        print('api whisper!')
        return transcribe_with_api(path_audio)
    except Exception as e:
        # Fall back to local model
        print(f"API error occurred: {e}. Falling back to local model.")
        print('whisper local!')
        return transcribe_with_local_model(path_audio)
