from torch.utils.data.dataset import Dataset, random_split
from gzip import GzipFile
import numpy as np 
import torch as th
import gym
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json
import matplotlib.pyplot as plt
import os
from gym import spaces
from math import exp, floor, ceil
from stable_baselines3 import PPO
from sb3_contrib.common.recurrent import RecurrentActorCritic
from gym.spaces import Box

class ExpertDataSet(Dataset):
    def __init__(self, expert_observations, expert_actions):
        self.observations = expert_observations # Song, 
        self.actions = expert_actions
        self.keys = list(expert_observations.keys())
        
    def __getitem__(self, index):
        return ({key: self.observations[key][index] for key in self.keys}, self.actions[index])

    def __len__(self): # Returns Num of Sequences
        return len(self.actions)
      
FILE_PATH = 'data/ExpertDataset/'
expert_observations = []
expert_actions = []
for file_idx in os.listdir(FILE_PATH):
  jnt_thetas = np.load(os.path.join(file_idx, 'joints.npy'), allow_pickle = True)
  actn_mtrx = np.load(os.path.join(file_idx, 'actn_mtrx.npy'), allow_pickle = True)
  sound = np.load(os.path.join(file_idx, 'sound.npy'), allow_pickle= True)
  
  expert_actions.append(np.load('actions.npy', allow_pickle = True))
  expert_observations.append({
    'joint_angles': jnt_thetas,
    'previous_actions': actn_mtrx,
    'music': sound
  })
  # Only the 3 fingers
  

expert_dataset = ExpertDataSet(expert_observations, expert_actions)

train_size = int(0.8 * len(expert_dataset))

test_size = len(expert_dataset) - train_size

train_expert_dataset, test_val_expert_dataset = random_split(
    expert_dataset, [train_size, test_size]
)

test_val_split = len(test_val_expert_dataset)/2
val_expert_dataset, test_expert_dataset = random_split(
    test_val_expert_dataset, [floor(test_val_split), ceil(test_val_split)]
)

print("test_expert_dataset: ", len(test_expert_dataset))
print("train_expert_dataset: ", len(train_expert_dataset))
print('val_expert_dataset: ', len(val_expert_dataset))

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
    batch_size = 120, 
    epochs = 1000, 
    learning_rate = 2e-5, 
    log_interval = 100, 
    no_cuda = False, 
    seed = 1, 
    patience = 10):

  use_cuda = not no_cuda and th.cuda.is_available()
  th.manual_seed(seed)
  device = th.device("cuda" if use_cuda else "cpu")
  kwargs = {"num_workers": 1, "pin_memory" : True} if use_cuda else {}

  criterion = nn.CrossEntropyLoss()
  
  # Extract
  model = student.policy.to(device)

  def train(model, device, train_loader, optimizer):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
      target = target.to(device)
      for k,v in data.items():
        data[k] = v.to(device)
      
      optimizer.zero_grad()


      dist = model.get_distribution(data)
      action_prediction = [i.logits for i in dist.distribution]
    #   action_prediction = [i.probs for i in dist.distribution]
      target = target.long()
      # CHANGE
      loss1 = criterion(action_prediction[0], target[:, 0])
      loss2 = criterion(action_prediction[1], target[:, 1])
      loss3 = criterion(action_prediction[2], target[:, 2])
      loss = loss1 + loss2 + loss3
      loss.backward()

      optimizer.step()
      if batch_idx % log_interval == 0:
        print('train epoch: {} [{}/{} ({:.0f}%)]\t loss:{:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100 * batch_idx / len(train_loader), loss.item()))

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

  train_loader = th.utils.data.DataLoader(dataset = train_expert_dataset, batch_size = batch_size, shuffle = True, **kwargs)
  test_loader = th.utils.data.DataLoader(dataset = test_expert_dataset, batch_size = batch_size, shuffle = True, **kwargs)
  val_loader = th.utils.data.DataLoader(dataset = val_expert_dataset, batch_size = batch_size, shuffle = True, **kwargs)


  optimizer = optim.AdamW(model.parameters(), lr = learning_rate)
  scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience-5, verbose=True)
#   scheduler = StepLR(optimizer, step_size = 1, gamma = scheduler_gamma)

  last_loss = 100
  trigger_times = 0
  for epoch in range(1, epochs+1):
    train(model, device, train_loader, optimizer)
    cur_loss = validation(model, device, val_loader)
    if last_loss < cur_loss:
        trigger_times += 1
        print('Last loss', last_loss)
        if trigger_times == patience:
            print('Early Stopping')
            break
    else:
        last_loss = cur_loss
        print('Updating Val_loss')
        trigger_times = 0
    test(model, device, test_loader)
    scheduler.step(cur_loss)

  student.policy = model

  student.save("ppo_model_multi_discrete")
  get_accuracy(student, test_expert_dataset, 'test')
  get_accuracy(student, val_expert_dataset, 'val')
  get_accuracy(student, train_expert_dataset, 'train')

pretrain_agent(
    ppo_model,
    epochs=2000,
    learning_rate= 3e-5,
    log_interval=100,
    no_cuda=False,
    seed=1,
    batch_size=128,
    patience = 15
)



# plt.plot(list(range(len(loss_series))), loss_series)
# plt.savefig('loss.jpg')