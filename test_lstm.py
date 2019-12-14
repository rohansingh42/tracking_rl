import matplotlib.pyplot as plt
#import gym          # Tested on version gym v. 0.14.0 and python v. 3.17
#########################################################################
#NN code
import numpy as np
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from torch.autograd import Variable 
from tensorboardX import SummaryWriter
from datetime import datetime 
import glob, os 
import argparse
from collections import deque
import pygame
import random
from Game import Game

pygame.init()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

game = Game()

# Define memory class for experience relay
class Memory():
    def __init__(self, memsize):
        self.memsize = memsize
        self.memory = deque(maxlen = self.memsize)
    def add_episode(self, episode):
        self.memory.append(episode)
    def gen_batch(self, bsize, time_step):
        sampled_episodes = random.sample(self.memory, bsize)
        batch = []
        for episode in sampled_episodes:
            # print(episode)
            point = np.random.randint(0, len(episode)+1-time_step)
            batch.append(episode[point:point+time_step])
        return batch

#################################################################
# Create Deep Recurrent Q Network (using LSTM)
#################################################################
# Define hyper parameters
INPUT_SIZE = 8
OUT_SIZE = 5
BATCH_SIZE = 32
TIME_STEP = 8
GAMMA = 0.99
INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.1
TOTAL_EPISODES = 20000
MAX_STEPS = 500
MEMORY_SIZE = 3000 # size of the list to store last n episodes
UPDATE_FREQ = 50
PERFORMANCE_SAVE_INTERVAL = 200 
TARGET_UPDATE_FREQ = 20000 # num of steps

# Define the network architecture
class Network(nn.Module):
    def __init__(self, input_size, out_size):
        super(Network, self).__init__()
        self.input_size = input_size
        self.output_size = out_size
        self.hidden_layer1 = 32
        self.hidden_layer2 = 64
        self.hidden_layer3 = 32
        # check how to include batches
        self.fc1 = nn.Linear(self.input_size, 
                             self.hidden_layer1)
        self.fc2 = nn.Linear(self.hidden_layer1, self.hidden_layer2)
        self.fc3 = nn.Linear(self.hidden_layer2, self.hidden_layer3)
        self.lstm_layer = nn.LSTM(input_size = 32, 
                                  hidden_size = 32,
                                  num_layers = 1, 
                                  batch_first = True) 
        self.adv = nn.Linear(32, self.output_size)
        self.val = nn.Linear(32, 1)

    def forward(self, x, bsize, time_step, hidden_state, cell_state):
        x = x.view(bsize*time_step,1, self.input_size)
        model = torch.nn.Sequential(self.fc1, 
                                    nn.ReLU(inplace=False),
                                    self.fc2,
                                    nn.ReLU(inplace=False),
                                    self.fc3,
                                    nn.ReLU(inplace=False))
        # out = self.fc1(x)
        # out = F.ReLU(out)
        # out = self.fc2(out)
        # out = F.ReLU(out)
        # out = self.fc3(out)
        # out = F.ReLU(out)
        out = model(x).view(bsize, time_step, 32)
        lstm_out = self.lstm_layer(out, (hidden_state, cell_state))
        out = lstm_out[0][:, time_step-1, :]
        h_n = lstm_out[1][0]
        c_n = lstm_out[1][1]

        adv_out = self.adv(out)
        val_out = self.val(out)

        Qout = val_out.expand(bsize, self.output_size) + (adv_out - adv_out.mean(dim=1).unsqueeze(dim=1).expand(bsize, self.output_size))
        
        return Qout, (h_n, c_n)
    
    def init_hidden_states(self, bsize):
        h = torch.zeros(1, bsize, 32).float().to(device)
        c = torch.zeros(1, bsize, 32).float().to(device)
        return h,c

# initialize networks
nn_model = Network(input_size = INPUT_SIZE, out_size = OUT_SIZE).to(device)
nn_model.load_state_dict(torch.load('LSTM_CKPT3.pth'), strict=False)
nn_model.eval()
print(nn_model)
target_model = Network(input_size = INPUT_SIZE, out_size = OUT_SIZE).to(device)

target_model.load_state_dict(nn_model.state_dict())
loss_fn = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(nn_model.parameters(), lr = 0.0001)
torch.manual_seed(1); np.random.seed(1)
path = glob.glob(os.path.expanduser('./logs/'))[0]
SummaryWriter = SummaryWriter('{}{}'.format(path, datetime.now().strftime('%b%d_%H-%M-%S')))

###################################################################
# Start algorithm
epsilon = INITIAL_EPSILON
loss_stat = []
reward_stat = []
total_steps = 0
n_success = 0

for episode in trange(0, TOTAL_EPISODES):
    curr_state = game.reset()
    episode_reward = 0
    step_count = 0
    episode_loss = 0
    local_memory = []

    hidden_state, cell_state = nn_model.init_hidden_states(bsize = 1)

    while step_count < MAX_STEPS:
        #render
        game.render()
        step_count += 1
        total_steps += 1

        torch_x = torch.from_numpy(np.asarray(curr_state)).float().to(device)
        model_out = nn_model.forward(torch_x, bsize=1, time_step=1,
                                        hidden_state = hidden_state,
                                        cell_state = cell_state)
        out = model_out[0]
        action = int(torch.argmax(out[0]))
        if action != 1:
            print(action)
        hidden_state = model_out[1][0]
        cell_state = model_out[1][1]

        # update model fro next iteration
        next_state, reward, done, success = game.step(action)
        episode_reward += reward
        curr_state = next_state

        if reward != -1:
            print(reward)
        # if epsilon > FINAL_EPSILON:
        #     epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/TOTAL_EPISODES 
        # SummaryWriter.add_scalar('data/episode_reward', episode_reward, episode)
        # SummaryWriter.add_scalar('data/episode_loss', episode_loss, episode)

        if done or success:
            if success:
                n_success += 1
                SummaryWriter.add_scalar('data/cumulative_success', n_success, episode)
                SummaryWriter.add_scalar('data/success', 1, episode)
            break
    # if episode % 200 == 0:
    #     torch.save(nn_model.state_dict(), 'LSTM_CKPT.pth')



