from src.Utils.data_utils import *

import os
import numpy as np
from collections import deque
from src.Utils.data_utils import one_hot_encode, get_mel_image_from_int32

DEG0F_F = 17
ACTION_SPACE = 11
DATA_PATH = os.path.join(os.getcwd(), 'data', 'SplicedData')
EXPERT_DATA_PATH = os.path.join(os.getcwd(), 'data', 'ExpertDataset')
ACTION_KEY = np.array([-60, -45, -30, -15, -5, 0, 5, 15, 30, 45, 60])

def convert_splice_to_expert():
    global DATA_PATH, EXPERT_DATA_PATH, ACTION_KEY
    actions_arr = []
    songs_arr = []
    joints_arr = []
    prev_act_matricies = []
    
    for song_idx in os.listdir(DATA_PATH):
        song_actions = []
        song_joints = []
        song_act_matricies = []
        mel_imgs = []
        pos_estimate_data = None
        song_audio = None
        try:
            pos_estimate_data = np.load(os.path.join(DATA_PATH, song_idx, 'FrameData.npy'))
            song_audio = np.load(os.path.join(DATA_PATH, song_idx, 'AudioData.npy'))
        except:
            continue
        actions_deq = deque()
        for _ in range(3):
            actions_deq.append(np.zeros(shape=(11*17)))
        
        for action_idx in range(len(pos_estimate_data)):
            d_thetas = None
            try:
                d_thetas = pos_estimate_data[action_idx+1] - pos_estimate_data[action_idx]
            except:
                d_thetas = np.zeros(DEG0F_F)
            song_joints.append(pos_estimate_data[action_idx] / 180)
            action = []
            
            for theta in range(len(d_thetas)):
                theta = d_thetas[theta]
                if theta > 180: # Decrease Theta
                    theta= 360 - theta
                elif theta < -180:
                    theta = -360 - theta
                argsub_actions = ACTION_KEY - theta
                action_idx = np.absolute(argsub_actions).argmin()
                action.append(action_idx)
            action = np.array(action)
            mel_imgs.append(get_mel_image_from_int32(song_audio[action_idx]))
            song_act_matricies.append(np.array(actions_deq).copy().reshape((3,DEG0F_F*ACTION_SPACE)))
            actions_deq.append(one_hot_encode(action))
            actions_deq.popleft()
            song_actions.append(action)
          
        actions_arr.append(song_actions)
        prev_act_matricies.append(song_act_matricies)
        songs_arr.append(mel_imgs)
        joints_arr.append(song_joints)
        print('Saving: ', song_idx)
    np.save(os.path.join(EXPERT_DATA_PATH, 'actions.npy'), actions_arr)
    np.save(os.path.join(EXPERT_DATA_PATH, 'joints.npy'), joints_arr)
    np.save(os.path.join(EXPERT_DATA_PATH, 'actn_mtrx.npy'), prev_act_matricies)
    np.save(os.path.join(EXPERT_DATA_PATH, 'sound.npy'), songs_arr)
            
            