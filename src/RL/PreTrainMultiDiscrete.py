from torch.utils.data.dataset import Dataset, random_split
from gzip import GzipFile
import numpy as np 
import torch as th
import gym
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.functional import one_hot
import json
import matplotlib.pyplot as plt
import os
from gym import spaces
from math import exp, floor, ceil
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from gym.spaces import Box

class ExpertDataSet:
    def __init__(self, expert_observations):
        print('Shape Actions: ', np.shape(expert_observations['actions'][0]))
        print('Shape Music: ', expert_observations['music'].shape)
        
        self.observations = expert_observations # Song, Sequence x Index]
        # self.keys = list(expert_observations.keys())
    # Returns a Sequence
    def __getitem__(self, sequence_num):
        return {'music': th.tensor(self.observations['music'][sequence_num]), 'joint_angles': th.tensor(self.observations['joint_angles'][sequence_num]), 'previous_actions': th.tensor(self.observations['previous_actions'][sequence_num]), 'actions': th.tensor(self.observations['actions'][sequence_num])}

    def __len__(self): # Returns Num of Sequences
        return len(self.observations['music'])
      
def get_expert_data():    
  FILE_PATH = 'data/ExpertDataset/'
  expert_observations = []
  jnt_thetas = np.load(os.path.join(FILE_PATH, 'joints.npy'), allow_pickle = True)
  actn_mtrx = np.load(os.path.join(FILE_PATH, 'actn_mtrx.npy'), allow_pickle = True)
  sound = np.load(os.path.join(FILE_PATH, 'sound.npy'), allow_pickle= True)  
  expert_actions= np.load(os.path.join(FILE_PATH, 'actions.npy'), allow_pickle = True)
  print('Shape Actions: ', np.shape(expert_actions[0]))
  expert_observations = {
      'joint_angles': jnt_thetas,
      'previous_actions': actn_mtrx,
      'music': sound,
      'actions': expert_actions
    }
  return ExpertDataSet(expert_observations)

  # Only the 3 fingers


# train_size = int(0.8 * len(expert_dataset))

# test_size = len(expert_dataset) - train_size

# train_expert_dataset, test_val_expert_dataset = random_split(
#     expert_dataset, [train_size, test_size]
# )

# test_val_split = len(test_val_expert_dataset)/2
# val_expert_dataset, test_expert_dataset = random_split(
#     test_val_expert_dataset, [floor(test_val_split), ceil(test_val_split)]
# )

# print("test_expert_dataset: ", len(test_expert_dataset))
# print("train_expert_dataset: ", len(train_expert_dataset))
# print('val_expert_dataset: ', len(val_expert_dataset))

def get_accuracy(model, data, dtype):
    count = 0
    total = 0
    for i in data:
      x = model.predict(i[0], deterministic = True)[0]
      if all(x == i[1]):
        count += 1
      total += 1
    print(dtype, count, total, count/total)

def pretrain_agent(
    student, 
    batch_size = 64, 
    epochs = 1000, 
    learning_rate = 3e-5, 
    log_interval = 5, 
    no_cuda = False, 
    seed = 1, 
    patience = 10):

  use_cuda = not no_cuda and th.cuda.is_available()
  th.manual_seed(seed)
  device = th.device("cuda" if use_cuda else "cpu")
  print(device)
  kwargs = {"num_workers": 1, "pin_memory" : True} if use_cuda else {}

  criterion = nn.CrossEntropyLoss()
  
  # Extract
  model = student.policy.to(device)

  def train(model, device, train_loader, optimizer):
    model.train()

    for sequence_idx in range(len(train_loader)):
          total_loss = None
          sequence = train_loader[sequence_idx]
          
          optimizer.zero_grad()
          
          lstm_states = (th.zeros(model.lstm_hidden_state_shape, device='cuda'), th.zeros(model.lstm_hidden_state_shape, device='cuda'))#, RNNStates(th.zeros(model.lstm_hidden_state_shape), th.zeros(model.lstm_hidden_state_shape))) # Batch Size x Features Dim x something... this code is not clear so I will have to investigate for future
          episode_starts = []
          train_idx = 0
          for seq_idx in range(sequence['actions'].shape[0]):
            observation, action = {'music': sequence['music'][seq_idx].unsqueeze(0).unsqueeze(0)  , 'joint_angles': sequence['joint_angles'][seq_idx].unsqueeze(0), 'previous_actions': sequence['previous_actions'][seq_idx].unsqueeze(0).unsqueeze(0)}, sequence['actions'][seq_idx].unsqueeze(0)
            for k,v in observation.items():
                observation[k] = v.to(device)
            action = action.to(device)
            episode_starts.append(0)
            dist, lstm_states = model.get_distribution(observation, lstm_states = lstm_states, episode_starts = th.tensor(episode_starts))
            action_prediction = [th.nn.functional.normalize(i.probs) for i in dist.distribution]
            target_one_hot = th.nn.functional.one_hot(action[0], num_classes=11).float()
    #   action_prediction = [i.probs for i in dist.distribution]
            # action = action.long()
            # CHANGE, might need target[] since we are going through each action in that batch
            loss = None
            for i in range(len(action_prediction)): 
              loss = criterion(action_prediction[i], target_one_hot[i].unsqueeze(0)) if loss is None else loss + criterion(action_prediction[i], target_one_hot[i].unsqueeze(0))
            loss /= len(action_prediction)
            total_loss = loss if total_loss is None else total_loss + loss
            print('Episodic Loss: ', loss.item())
            train_idx+=1
            if train_idx % batch_size == 0:
              total_loss/= batch_size
              total_loss.backward()
              total_loss = None
              optimizer.step()
              lstm_states = (lstm_states[0].clone().detach().to(device), lstm_states[1].clone().detach().to(device))
              optimizer.zero_grad()
              
          if total_loss:  
            total_loss /= (sequence['actions'].shape[0] % batch_size)
            total_loss.backward()
            optimizer.step()
            print('Total Episodic Loss: ', total_loss.item())
          
          
    if sequence_idx % log_interval == 0:
      print('train epoch: {} Sequence: {} ({:.0f}%)]\t loss:{:.6f}'.format(epoch, sequence_idx, total_loss.item()))

  def validation(model, device, val_loader):
    model.eval()
    loss_total = 0
    with th.no_grad():
      for data, target in val_loader:
        target = target.to(device)
        for k, v in data.items():
          data[k] = v.to(device)

        dist = model.get_distribution(data)
        action_prediction = [i.logits for i in dist.distribution]
        #   action_prediction = [i.probs for i in dist.distribution]
        target = target.long()

        loss1 = criterion(action_prediction[0], target[:, 0])
        loss2 = criterion(action_prediction[1], target[:, 1])
        loss3 = criterion(action_prediction[2], target[:, 2])
        val_loss = loss1 + loss2 + loss3
        loss_total += val_loss.item()

    val_loss = loss_total / len(val_loader.dataset)
    print('Validation_loss:', val_loss)
    return val_loss


  def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    with th.no_grad():
      for data, target in test_loader:
        target = target.to(device)
        for k, v in data.items():
          data[k] = v.to(device)

        dist = model.get_distribution(data)
        action_prediction = [i.logits for i in dist.distribution]
        #   action_prediction = [i.probs for i in dist.distribution]
        target = target.long()

        loss1 = criterion(action_prediction[0], target[:, 0])
        loss2 = criterion(action_prediction[1], target[:, 1])
        loss3 = criterion(action_prediction[2], target[:, 2])
        loss = loss1 + loss2 + loss3

        test_loss += loss.item()

    test_loss = test_loss / len(test_loader.dataset)
    # print('Validation_loss:', val_loss)
        # test_loss = criterion(action_prediction, target)
    # test_loss /= len(test_loader.dataset)
    print(f'test set: average loss: {test_loss:.4f}')

  train_loader = get_expert_data()
  # test_loader = th.utils.data.DataLoader(dataset = test_expert_dataset, batch_size = batch_size, shuffle = True, **kwargs)
  # val_loader = th.utils.data.DataLoader(dataset = val_expert_dataset, batch_size = batch_size, shuffle = True, **kwargs)


  optimizer = optim.Adam(model.parameters(), lr = learning_rate,
        eps=1e-5,
        weight_decay=0,)
  # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience-5, verbose=True)
#   scheduler = StepLR(optimizer, step_size = 1, gamma = scheduler_gamma)

  # last_loss = 100
  # trigger_times = 0
  print('Training...')
  for epoch in range(1, epochs+1):
    train(model, device, train_loader, optimizer)
  #   # cur_loss = validation(model, device, val_loader)
  #   # if last_loss < cur_loss:
  #       # trigger_times += 1
  #       # print('Last loss', last_loss)
  #       # if trigger_times == patience:
  #           # print('Early Stopping')
  #           # break
  #   # else:
  #       # last_loss = cur_loss
  #       # print('Updating Val_loss')
  #       # trigger_times = 0
  #   # test(model, device, test_loader)
  #   # scheduler.step(cur_loss)

  # student.policy = model

  # student.save("ppo_model_multi_discrete")
  # get_accuracy(student, test_expert_dataset, 'test')
  # get_accuracy(student, val_expert_dataset, 'val')
  # get_accuracy(student, train_expert_dataset, 'train')

# pretrain_agent(
#     ppo_model,
#     epochs=2000,
#     learning_rate= 3e-5,
#     log_interval=100,
#     no_cuda=False,
#     seed=1,
#     batch_size=128,
#     patience = 15
# )



# plt.plot(list(range(len(loss_series))), loss_series)
# plt.savefig('loss.jpg')