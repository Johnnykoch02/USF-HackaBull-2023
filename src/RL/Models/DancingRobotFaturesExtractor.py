from stable_baselines3 import PPO
import os
from typing import Dict, List
import gym
from gym import spaces

from torch import nn
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class DancingRobotFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super(DancingRobotFeaturesExtractor, self).__init__(observation_space, features_dim= 1)
        extractors = {}
        total_concat_size = 0
                    
        for key, subspace in observation_space.spaces.items():
                if key == 'joint_angles':
                    extractors[key] = nn.Sequential(
                        nn.Linear(17, 120),
                        nn.LeakyReLU(),
                        nn.BatchNorm1d(120),
                        nn.Linear(120, 80),
                        nn.LeakyReLU(),
                    )
                    total_concat_size+=80
                elif key == 'previous_actions':
                    extractors[key] = nn.Sequential(
                        nn.BatchNorm2d(1),
                        nn.Conv2d(1, 32, kernel_size=(11,3), stride=11, padding='same'),
                        nn.LeakyReLU(),
                        nn.BatchNorm2d(32),
                        nn.Conv2d(32, 128, kernel_size=(11,3), stride=11, padding='same'), 
                        nn.LeakyReLU(),
                        nn.MaxPool2d((2, 2)),
                        nn.Flatten(),
                    )#TODO: Calculate the CONCAT 
                    total_concat_size+=128
                elif  'music' in key:
                    extractors[key] = nn.Sequential(
                        nn.BatchNorm2d(subspace.shape[0]),
                        nn.Conv2d(subspace.shape[0], 32, kernel_size=(4, 4), stride=2, padding='same'),
                        nn.LeakyReLU(),
                        nn.Conv2d(32, 64, (3,3), padding='same'),
                        nn.LeakyReLU(),
                        nn.Conv2d(64, 128, (3, 3), padding= 'same'),
                        nn.LeakyReLU(),
                        nn.AdaptiveAvgPool2d((1,1)),
                        nn.Flatten(),  
                    )
                    total_concat_size+=128
   
        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size
        self.output_dim = total_concat_size
    
    def forward(self, observations):
        encoded_tensor_list = []
        '''extractors contain nn.Modules that do all of our processing '''
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
    
        return th.cat(encoded_tensor_list, dim= 1)