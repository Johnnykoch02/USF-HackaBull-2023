print('Go World')
import os
import sys
# os.environ["CUDA_DEVICE_ORDER"] ="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# from stable_baselines3.common import env_checker
# from src.RL.callbacks import TrainAndLoggingCallback 
from src.env.DancingRobotGym import DancingRobotGym, NAME, CHECKPOINT_DIR, LOG_DIR
from src.RL.Models.DancingRobotFaturesExtractor import DancingRobotFeaturesExtractor
from src.RL.PreTrainMultiDiscrete import pretrain_agent
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib import RecurrentPPO
from src.DataExtractors.ActionExtractor import convert_splice_to_expert

# https://npitsillos.github.io/blog/2021/recurrent-ppo/ This Talks about Recurrent PPO

# PPO.load(os.path.join(CHECKPOINT_DIR, 'Exploration_Model.zip'))

import threading as t
import time
import numpy as np
import torch as th

factor = 0.99999
stop_factor_lr = 5e-5
lr = 0.001

MFCC = (28, 40)

def lr_call(step):
    global lr, stop_factor_lr, factor
    lr = max(stop_factor_lr, lr * factor)
    return lr

# TODO: LR Scheduler

policy_kwargs = dict(
    features_extractor_class=DancingRobotFeaturesExtractor,# pi: action, vf: value
    # net_arch=[512, 126, dict(pi=[128,64], vf=[128,64])],
    optimizer_class=th.optim.RMSprop, #Adam    
    optimizer_kwargs=dict(
        alpha=1.0,
        eps=1e-5,
        weight_decay=0,
    ),
)

def extract_command(command:str):
    cmd = command.split(' ')[0]
    arguments = command.replace(cmd,'').split(' -')
    args = {}
    for i in arguments:
        try:
            tmp = i.replace(' ', '#',1).split('#')
            args[tmp[0]] = tmp[1].strip()
        except:
            pass
    return cmd, args

# def Demo(model_loc):
#     model = PPO.load(os.path.join(CHECKPOINT_DIR, 'Exploration_Model.zip'))
#     env = BerrettHandGym()
#     env.configs = np.load(os.path.join(os.getcwd(), "Data_Collection", "Difficultys"," easy.npy"), allow_pickle= True)
#     model.set_env(env)
#     obs = env.reset()
#     i = 0
#     reward = []
#     Dones = []
#     done = False
#     while i < 10000:
#         if done:
#             Dones.append(done)
#             env.reset()
#         i+=1
#         action, _states = model.predict(obs.copy())
#         obs, rewards, done, info = env.step(action)
#         reward.append(rewards)
#         if i % 60 == 0:
#             print("Percent Success: {:.2f}%".format((sum(Dones)/ len(Dones)) * 100))
#             print('Mean Rew. for Past 60 timesteps:', np.mean(reward[-60::]),'Demo Info:', info )

'''train -num_steps <int> -load_dir <path>'''
def Train(num_steps:int=100000, load_dir:str='', detailed_training=False):
    model = None
    if load_dir != '':
        model = RecurrentPPO.load(os.path.join(os.getcwd(), CHECKPOINT_DIR, load_dir), tensorboard_log=LOG_DIR)
        model.set_env(DancingRobotGym())

    else:
        model = RecurrentPPO(
            'MultiInputLstmPolicy',
            env=DancingRobotGym(MFCC),
            policy_kwargs=policy_kwargs, 
            verbose = 1,
            learning_rate = 7e-4,
            gae_lambda=1,
            normalize_advantage=False,
            n_epochs=1,
            clip_range_vf=None,
            
            n_steps = 256,
            batch_size = 256,
            
            ent_coef = 0.001,
            tensorboard_log=LOG_DIR)
        
    model.learn(total_timesteps=100000)
    # detected_bad_scenes = model.env.bad_scenes
    model.save(os.path.join(os.path.join(os.getcwd(), CHECKPOINT_DIR, f'Pretrained_{NAME}.zip')))

def PreTrain(num_steps:int=100000, load_dir:str='', detailed_training=False):
    model = None
    if load_dir != '':
        model = RecurrentPPO.load(os.path.join(os.getcwd(), CHECKPOINT_DIR, load_dir), tensorboard_log=LOG_DIR)
        model.set_env(DancingRobotGym())

    else:
        model = RecurrentPPO(
            'MultiInputLstmPolicy',
            env=DancingRobotGym(MFCC),
            policy_kwargs=policy_kwargs, 
            verbose = 1,
            learning_rate = 7e-4,
            gae_lambda=1,
            normalize_advantage=False,
            n_epochs=1,
            clip_range_vf=None,
            
            n_steps = 256,
            batch_size = 256,
            
            ent_coef = 0.001,
            tensorboard_log=LOG_DIR)
        
        
    pretrain_agent(model, epochs=2000, batch_size=1, patience=15)
    # detected_bad_scenes = model.env.bad_scenes
    model.save(os.path.join(os.path.join(os.getcwd(), CHECKPOINT_DIR, f'Pretrained_{NAME}.zip')))
    
    '''We can choose what to do with this in the future'''
    
    # print(f"Final Accuracy: {(sum(model.env.successes[-250::])/1000):.2%}")
        
commands = {'exit': {'fnc': sys.exit , 'args':['load_dir', 'num_steps']},
            'train': {'fnc': Train , 'args':['load_dir', 'num_steps']},
           # 'demo': {'fnc': Demo, 'args':['model_loc']},
            #'data_collect': {'fnc': DataCollect,' args': ['num_steps']}
            }
def main():
    global commands
    # Train(num_steps=1000, detailed_training=True)
    # convert_splice_to_expert()
    Train()


if __name__ == '__main__':
    main()
    
