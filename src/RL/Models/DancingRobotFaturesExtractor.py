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
                        # nn.BatchNorm1d(120),
                        nn.Linear(120, 80),
                        nn.LeakyReLU(),
                    )
                    total_concat_size+=80
                elif key == 'previous_actions':
                    extractors[key] = nn.Sequential(
                        nn.BatchNorm2d(1),
                        nn.Conv2d(1, 32, kernel_size=(3, 11), stride=11,),
                        nn.LeakyReLU(),
                        nn.BatchNorm2d(32),
                        nn.Flatten(),
                    )#TODO: Calculate the CONCAT 
                    total_concat_size+=544
                elif  'music' in key:
                    extractors[key] = nn.Sequential(
                        nn.BatchNorm2d(1),
                        nn.Conv2d(1, 32, kernel_size=(7, 8), padding='same'),
                        nn.LeakyReLU(),
                        nn.Conv2d(32, 64, kernel_size=(3,3), padding='same'),
                        nn.LeakyReLU(),
                        nn.BatchNorm2d(64),
                        nn.Conv2d(64, 128, kernel_size=(3, 3)),
                        nn.LeakyReLU(),
                        nn.MaxPool2d((2, 2)),
                        nn.Conv2d(128, 256, kernel_size=(3, 3)),
                        nn.LeakyReLU(),
                        nn.BatchNorm2d(256),
                        nn.MaxPool2d((4, 4)),
                        nn.Flatten(),  
                    )
                    total_concat_size+=2048
   
        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size
        self.output_dim = total_concat_size
    
    def forward(self, observations):
        encoded_tensor_list = []
        '''extractors contain nn.Modules that do all of our processing '''
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
            continue
    
        return th.cat(encoded_tensor_list, dim= 1)