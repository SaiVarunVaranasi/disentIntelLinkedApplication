import os
import matplotlib.pyplot as plt
import numpy as np
import torch as t
from pathlib import Path
import librosa
import librosa.display
from librosa.sequence import dtw
import warnings
warnings.filterwarnings('ignore')

from inferencer import Inferencer
from assets.py_vad_tool.py_vad_tool.audio_tools import * 
from assets.py_vad_tool.py_vad_tool.unsupervised_vad import compute_log_nrg, nrg_vad


def cut_vad_wav(
    file_path: Path() = None,
    c: float = 0.1,
    percent_thr=0.3,
    nrg_thr: float = 0.0,
    context: int = 20
):
        """
        Cut and VAD a single file, given by path.
        Returns the wav frames after preprocessing.

        :params file_path: path to the wav file
        :params c: percet to cut at the beginning and end of file previos to VAD
        :params percent_thr: VAD related parameter
        :params nrg_thr: VAD related parameter
        :params context: VAD related parameter - context to consider for VAD algo
        """
        assert os.path.exists(file_path)
        
        fs, s = read_wav(file_path)
        # cut c at start and end
        duration = len(s) / fs 
        cut_p = duration * c
        cut_p_frames = int(cut_p * fs)
        cut_start = cut_p_frames
        cut_end = len(s) - cut_p_frames
        s = s[cut_start:cut_end]
        # vad
        win_len = int(fs*0.025)
        hop_len = int(fs*0.010)
        sframes = enframe(s, win_len, hop_len)
        vad = nrg_vad(xframes=sframes, percent_thr=percent_thr, nrg_thr=nrg_thr, context=context)
        energy_frames = deframe(vad, win_len, hop_len)
        # create new audio file, with only selected energy frames
        new_frames = []
        for i in range(len(s)):
                if energy_frames[i] == 1.:
                        new_frames.append(s[i])
        return new_frames, fs


def wav_to_codes(
        wav_path: Path(),
        gender: str = '',
        weights_path: Path() = 'assets/weights/en/805000-G.ckpt',
        spk_emb_dim: int = 14212,
):
    """
    Returns a dict containing codes (Zc: content, Zr: rhythm, Zf: pitch),
    as well as the mspec and mspec without timbre information for a given .wav file.

    :params wav_path: path to the wav file from which codes should be extracted
    :params gender: gender of the speaker of wav_path - has to be 'F' or 'M'
    :params weights_path: path to the trained SpeechSplit model weights
    :params spk_emb_dim: has to match to the trained model weights (i.e. weights_path)
    """
    assert os.path.exists(wav_path)
    file_name_base = wav_path.split('/')[-1][:-4]
    assert os.path.exists(weights_path)
    assert gender in ['M', 'F'], 'Gender has to be M or F.'

    # preprocessing of the wav file (here cutting and Voice Activity Detection)
    prepro_frames, sr = cut_vad_wav(wav_path)

    # speechsplit output
    infer = Inferencer(weights_path)
    # mspec, f0_norm = infer.load_sample(wav_path, spk_emb_dim, gender)
    mspec, f0_norm = infer.mel_f0(prepro_frames, sr, spk_emb_dim, gender)
    mspec_pad, f0_norm_pad_onehot = infer.prepare_data(mspec, f0_norm)
    inf_out = infer.inference(mspec_pad, f0_norm_pad_onehot, spk_emb_dim)

    # 3 codes + mspec without timbre
    mel_length = mspec.shape[0]
    g_mel_noT, zc, zr, zf = inf_out
    g_mel_noT = g_mel_noT.squeeze().T[:, :mel_length]
    zc = zc.squeeze().T[:, :mel_length]
    zr = zr.squeeze().T[:, :mel_length]
    zf = zf.squeeze().T[:, :mel_length]

    # correct types/shapes
    mspec = mspec.T
    g_mel_noT = g_mel_noT.cpu().detach().numpy()
    zc = zc.cpu().detach().numpy()
    zr = zr.cpu().detach().numpy()
    zf = zf.cpu().detach().numpy()

    return {
        'mspec': mspec,
        'mspec_noT': g_mel_noT,
        'zc': zc,
        'zr': zr,
        'zf': zf
    }



def code_pair_diff(ref_code, pat_code):
    """
    Aliges the ref_code and the patho_code in the time domain using DTW.
    Make sure that ref_code and pat_code are the same code.

    :params ref_code: reference code
    :params patho_code: pathological code (will be aligned to ref_code)
    """
    # DTW
    dist, path = dtw(ref_code, pat_code)
    pat_code_aligned = np.empty(shape=(ref_code.shape), dtype=np.float32)
    for i_ref, i_pat in path:
        pat_code_aligned[:, i_ref] = pat_code[:, i_pat]

    # Difference (absolute square)
    code_diff = np.square(ref_code - pat_code_aligned)

    return code_diff, pat_code_aligned


def plot(code, ref_codes, pat_codes, pat_code_aligned, diff):
    """
    IMPORTANT: code and diff have to use the same underlying code (e.g. zc)
    
    :params code: the code that should be plottet
    :params ref_codes: dict output from wav_to_codes (healthy reference wav)
    :params pat_codes: dict output from wav_codes (pathological wav)
    :params diff: output from code_pair_diff
    """
    assert code in ['mspec', 'mspec_noT', 'zc', 'zr', 'zf']

    fig, (ax1, ax2, ax3) = plt.subplots(3, 2, constrained_layout=True, sharey='row')
    # ref
    ax1[0].set_title('mspec (Ref)', color='grey')
    librosa.display.specshow(ref_codes['mspec'], x_axis='frames', y_axis='hz', ax=ax1[0])
    ax2[0].set_title(code + ' (Ref)', color='grey')
    im1 = ax2[0].imshow(ref_codes[code], aspect='auto', origin='lower')

    # pat
    ax1[1].set_title('mspec (Pat)', color='grey')
    librosa.display.specshow(pat_codes['mspec'], x_axis='frames', y_axis='hz', ax=ax1[1])
    ax2[1].set_title(code + ' aligned (Pat)', color='grey')
    im2 = ax2[1].imshow(pat_code_aligned, aspect='auto', origin='lower')

    # diff
    ax3[0].set_title(code + ' aligned diff', color='grey')
    im3 = ax3[0].imshow(diff, aspect='auto', origin='lower')

    plt.show()
    plt.close(fig)

    





if __name__ == '__main__':
    # Example
    ref_wav = './example/CF02_B1_C1_M2.wav'
    patho_wav = './example/F02_B1_C1_M2.wav'

    ref_codes = wav_to_codes(ref_wav, 'F')
    pat_codes = wav_to_codes(patho_wav, 'F')

    zc_diff, pat_zc_aligned = code_pair_diff(ref_codes['zc'], pat_codes['zc'])
    zr_diff, pat_zr_aligned = code_pair_diff(ref_codes['zr'], pat_codes['zr'])
    zf_diff, pat_zf_aligned = code_pair_diff(ref_codes['zf'], pat_codes['zf'])

    plot('zc', ref_codes, pat_codes, pat_zc_aligned, zc_diff)






