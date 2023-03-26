import librosa
import os
import numpy as np
from tqdm import tqdm

def get_mel_image_from_int32(audio_32, max_len=64, n_mfcc=28):
    audio = audio.astype(np.float32, order='C') / 32768.0
    # wave = audio[::3]
    mfcc = librosa.feature.mfcc(y=audio, sr=48000, n_mfcc=n_mfcc)
    if (max_len > mfcc.shape[1]):
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc