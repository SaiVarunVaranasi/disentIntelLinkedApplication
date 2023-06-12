import os
import argparse
import torch
from torch.backends import cudnn
from fastapi import FastAPI
from pydantic import BaseModel
import wave
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import JSONResponse
from disentIntel import code_pair_diff, cut_vad_wav, wav_to_codes

from solver import Solver
from data_loader import get_loader
from hparams import hparams, hparams_debug_string


class SpeechData(BaseModel):
    field1: str
    field2: str


def str2bool(v):
    return v.lower() in ('true')


def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)

    # Data loader.
    vcc_loader = get_loader(hparams)

    # Solver for training
    solver = Solver(vcc_loader, config, hparams)

    solver.train()


app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "DisentIntel Algorithm"}


@app.post("/analyze")
def your_function(data: SpeechData):
    c_value = 0.2
    percent_thr_value = 0.5
    nrg_thr_value = 0.1
    context_value = 30

    # recordedData = data.field2.read()
    # wav_file_path_RcrdDta = "./example/CF02_B1_C1_M2.wav"
    # convertBlobtoWavFile(wav_file_path_RcrdDta, recordedData)
    # processed_RcrdDta = cut_vad_wav(
    #     wav_file_path_RcrdDta,
    #     c_value=c_value,
    #     percent_thr=percent_thr_value,
    #     nrg_thr=nrg_thr_value,
    #     context=context_value
    # )
    ref_wav = './example/CF02_B1_C1_M2.wav'
    patho_wav = './example/F02_B1_C1_M2.wav'

    ref_codes = wav_to_codes(ref_wav, 'F')
    pat_codes = wav_to_codes(patho_wav, 'F')

    zc_diff, pat_zc_aligned = code_pair_diff(ref_codes['zc'], pat_codes['zc'])
    zr_diff, pat_zr_aligned = code_pair_diff(ref_codes['zr'], pat_codes['zr'])
    zf_diff, pat_zf_aligned = code_pair_diff(ref_codes['zf'], pat_codes['zf'])

    plot('zc', ref_codes, pat_codes, pat_zc_aligned, zc_diff)

    response = {"message": "API endpoint successfully called"}
    return response


def convertBlobtoWavFile(wav_file_path, audio_data):
    with wave.open(wav_file_path, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Set the number of channels (1 for mono, 2 for stereo)
        wav_file.setsampwidth(2)  # Set the sample width in bytes (e.g., 2 bytes for 16-bit audio)
        wav_file.setframerate(44100)  # Set the sample rate (e.g., 44100 Hz)

        wav_file.writeframes(audio_data)


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Your API Title",
        version="1.0.0",
        description="Your API description",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema


@app.get("/docs", include_in_schema=False)
async def swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="Your API documentation"
    )


@app.get("/openapi.json", include_in_schema=False)
async def get_openapi_endpoint():
    return JSONResponse(custom_openapi())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Training configuration.
    parser.add_argument('--num_iters', type=int, default=1000000, help='number of total iterations')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')

    # Miscellaneous.
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)
    parser.add_argument('--device_id', type=int, default=0)

    # Directories.
    parser.add_argument('--log_dir', type=str, default='run/logs')
    parser.add_argument('--model_save_dir', type=str, default='run/models')
    parser.add_argument('--sample_dir', type=str, default='run/samples')

    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=1000)  # set to 1 for debugging on local machine
    parser.add_argument('--model_save_step', type=int, default=1000)
    parser.add_argument('--audio-step', type=int, default=75000)

    config = parser.parse_args()
    print(config)
    print(hparams_debug_string())
    main(config)
