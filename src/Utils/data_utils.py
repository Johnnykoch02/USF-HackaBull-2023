import librosa
import os
import numpy as np
from tqdm import tqdm
DEG0F_F = 17
ACTION_SPACE = 11

def get_mel_image_from_int32(audio_32, max_len=40, n_mfcc=28):
    audio = audio_32.astype(np.float32, order='C') / 32768.0
    # wave = audio[::3]
    mfcc = librosa.feature.mfcc(y=audio, sr=48000, n_mfcc=n_mfcc)
    if (max_len > mfcc.shape[1]):
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc

def one_hot_encode(action):#[0-14, 0-14, 0-14]
    global DEG0F_F, ACTION_SPACE
    encoded_action = np.zeros(shape=(DEG0F_F*ACTION_SPACE,))
    for i in range(len(action)):
        offset = (i*ACTION_SPACE)
        action_encode_index = action[i] + offset
        encoded_action[int(action_encode_index)] = 1
    return encoded_action