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

# Load game env
# env = gameEnv()
# curr_state = env.reset()

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
MAX_STEPS = 50
MEMORY_SIZE = 3000 # size of the list to store last n episodes
UPDATE_FREQ = 5
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
        self.fc2 = nn.Linear(self.fc1, self.fc2)
        self.fc3 = nn.Linear(self.fc2, self.fc3)
        self.lstm_layer = nn.LSTM(input_size = 32, 
                                  hidden_size = 32,
                                  num_layers = 1, 
                                  batch_first = True) 
        self.adv = nn.Linear(32, self.output_size)
        self.val = nn.Linear(32, 1)

    def forward(self, x, bsize, time_step, hidden_state, cell_state):
        x = x.view(bsize*time_step, self.input_size)
        model = torch.nn.Sequential(self.fc1, 
                                    nn.ReLU(inplace=False),
                                    self.fc2,
                                    nn.ReLU(inplace=False),
                                    self.fc3,
                                    nn.ReLU(inplace=False))
        lstm_out = (model(x), (hidden_state, cell_state))
        out = lstm_out[0][:, time_step-1, :]
        h_n = lstm_out[1][0]
        c_n = lstm_out[1][1]

        adv_out = self.adv(out)
        val_out = self.val(out)

        Qout = val_out.expand(bsize, self.output_size) + 
               (adv_out - adv_out.mean(dim=1).unsqueeze(dim=1).expand(bsize, self.output_size))
        
        return Qout, (h_n, c_n)
    
    def init_hidden_states(self, bsize):
        h = torch.zeros(1, bsize, 32).float().to(device)
        c = torch.zeros(1, bsize, 32).float().to(device)
        return h,c

# initialize networks
nn_model = Network(input_size = INPUT_SIZE, out_size = OUT_SIZE).to(device)
print(nn_model)
target_model = Network(input_size = INPUT_SIZE, out_size = OUT_SIZE).to(device)

target_model.load_state_dict(nn_model.state_dict())
loss_fn = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(nn_model.parameters(), lr = 0.0001)

# initialize memory
mem = Memory(memsize = MEMORY_SIZE)
# Fill the memory for the initial state
for i in range(0, MEMORY_SIZE):
    curr_state = env.reset()
    step_count = 0
    local_memory = []

    while step_count < MAX_STEPS:
        step_count += 1
        action = np.random.randint(0,5)
        next_state, reward, done = env.step(action)
        local_memory.append(curr_state, action, reward, next_state)
        curr_state = next_state
    mem.add_episode(local_memory)

print('Populated with %d Episodes'%(len(mem.memory)))

###################################################################
# Start algorithm
epsilon = INITIAL_EPSILON
loss_stat = []
reward_stat = []
total_steps = 0

for episode in range(0, TOTAL_EPISODES):
    curr_state = env.reset()
    episode_reward = 0
    step_count = 0
    episode_loss = 0
    local_memory = []

    hidden_state, cell_state = nn_model.init_hidden_states(bsize = 1)

    while step_count < MAX_STEPS:
        step_count += 1
        total_steps += 1

        if np.random.rand(1) < epsilon:
            torch_x = torch.from_numpy(curr_state).float().to(device)
            model_out = nn_model.forward(torch_x, bsize = 1, time_step=1,
                                         hidden_state = hidden_state, 
                                         cell_state = cell_state)
            action = np.random.randint(0,5)
            hidden_state = model_out[1][0]
            cell_state = model_out[1][1]
        else:
            torch_x = torch.from_numpy(curr_state).float().to(device)
            model_out = nn_model.forward(torch_x, bsize=1, time_step=1,
                                         hidden_state = hidden_state,
                                         cell_state = cell_state)
            out = model_out[0]
            action = int(torch.argmax(out[0]))
            hidden_state = model_out[1][0]
            cell_state = model_out[1][1]

        # update model fro next iteration
        next_state, reward, done = env.step(action)
        episode_reward += reward

        local_memory.append((curr_state, action, reward, next_state))
        curr_state = next_state

        # update target_model
        if(total_steps % TARGET_UPDATE_FREQ) == 0:
            target_model.load_state_dict(nn_model.state_dict())
        if(total_steps % UPDATE_FREQ) == 0:
            hidden_batch, cell_batch = nn_model.init_hidden_states(bsize = BATCH_SIZE)
            batch = mem.gen_batch(bsize = BATCH_SIZE, time_step = TIME_STEP)

            current_states = []
            actions = []
            rewards = []
            next_states = []

            for b in batch:
                cs, ac, rw, ns = [],[],[],[]
                for element in b:
                    cs.append(element[0])
                    ac.append(element[1])
                    rw.append(element[2])
                    ns.append(element[3])
                current_states.append(cs)
                actions.append(ac)
                rewards.append(rw)
                next_states.append(ns)

            current_states = np.array(current_states)
            actions = np.array(actions)
            next_states = np.array(next_states)
            rewards = np.array(rewards)

            torch_cs = torch.from_numpy(current_states).float().to(device)
            torch_ac = torch.from_numpy(actions).float().to(device)
            torch_rw = torch.from_numpy(rewards).float().to(device)
            torch_ns = torch.from_numpy(next_states).float().to(device)

            Q_next,_ = target_model.forward(torch_ns, bsize = BATCH_SIZE, time_step=TIME_STEP, hidden_state = hidden_batch, cell_state = cell_batch)
            Q_next_max,_ = Q_next.detach().max(dim=1)
            
            target_values = torch_rw[:,TIME_STEP-1] + (GAMMA * Q_next_max)

            Q_s, _ = nn_model.forward(torch_cs, bsize = BATCH_SIZE, time_step = TIME_STEP, hidden_state = hidden_batch, cell_state = cell_batch)

            Q_s_a, _ = Q_s.gather(dim=1, index = torch_ac[:,TIME_STEP-1].unsqueeze(dim=1)).squeeze(dim=1)

            loss = loss_fn(Q_s_a, target_values)

            # save performance measure
            loss_stat.append(loss.item())
            # make previous gradient zero
            optimizer.zero_grad()
            # update params
            optimizer.step()

            # Record history
            episode_reward += reward
            episode_loss += loss.item()
        mem.add_episode(local_memory)
        if epsilon > FINAL_EPSILON:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/TOTAL_EPISODES 
        SummaryWriter.add_scalar('data/episode_reward', episode_reward, episode)
        SummaryWriter.add_scalar('data/episode_loss', episode_loss, episode)
torch.save(nn_model.state_dict(), 'LSTM_CKPT.pth')



