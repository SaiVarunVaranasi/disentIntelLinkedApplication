from fastapi import FastAPI
from pydantic import BaseModel
import wave
import sys
sys.path.append('./Algorithm')
from disentIntel import cut_vad_wav



class SpeechData(BaseModel):
    field1: str
    field2: str

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "DisentIntel Algorithm"}

@app.post("/analyze")
def your_function(data: SpeechData):
    c_value = 0.2
    percent_thr_value = 0.5
    nrg_thr_value = 0.1
    context_value = 30
    healthyData = data.field1.read()
    wav_file_path_hltyDta = "Algorithm/files/healthyOutput.wav"
    convertBlobtoWavFile(wav_file_path_hltyDta, healthyData)
    processed_hltyDta = cut_vad_wav(wav_file_path_hltyDta,c=c_value,
    percent_thr=percent_thr_value,
    nrg_thr=nrg_thr_value,
    context=context_value)

    recordedData = data.field2.read()
    wav_file_path_RcrdDta = "Algorithm/files/recordedOutput.wav"
    convertBlobtoWavFile(wav_file_path_RcrdDta, recordedData)
    processed_RcrdDta = cut_vad_wav(wav_file_path_RcrdDtac=c_value,
    percent_thr=percent_thr_value,
    nrg_thr=nrg_thr_value,
    context=context_value)

    response = {"message": "API endpoint successfully called"}
    return response


def convertBlobtoWavFile(wav_file_path, audio_data):
    with wave.open(wav_file_path, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Set the number of channels (1 for mono, 2 for stereo)
        wav_file.setsampwidth(2)  # Set the sample width in bytes (e.g., 2 bytes for 16-bit audio)
        wav_file.setframerate(44100)  # Set the sample rate (e.g., 44100 Hz)

        wav_file.writeframes(audio_data)


