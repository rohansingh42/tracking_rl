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

import pygame
from Game import Game

pygame.init()

# Define Neural Net model
def weights_init(m):
	classname = m.__class__.name__
	if classname.find('Linear') != -1:
		nn.init.normal_(m.weight, 0, 1)

class model_network(nn.Module):
	def __init__(self):
		super(model_network, self).__init__()
		# update with env variable using pygame
		self.state_space = 8 # env.observation_space.shape[0]
		# update with env variable using pygame
		self.action_space = 5 # env.action_space.n
		self.hidden_layer1 = 32
		self.hidden_layer2 = 64
		self.hidden_layer3 = 32
		# define layers
		self.fc1 = nn.Linear(self.state_space, self.hidden_layer1, bias=True)
		self.fc2 = nn.Linear(self.hidden_layer1,self.hidden_layer2, bias=True)
		self.fc3 = nn.Linear(self.hidden_layer2, self.hidden_layer3, bias=True)
		
		self.output = nn.Linear(self.hidden_layer3, self.action_space, bias=True)

	def forward(self, x):
		model = torch.nn.Sequential(self.fc1, 
									nn.ReLU(inplace=False), 
									self.fc2, 
									nn.ReLU(inplace=False), 
									self.fc3, 
									nn.ReLU(inplace=False), 
									self.output)
		return model(x)

# Read user arguments
parser = argparse.ArgumentParser()
parser.add_argument("--train", default=True, help="Flag to Train or test the network")
parser.add_argument("--load_model", default=True, help="load pretrained model")
args = parser.parse_args()
train = args.train
if str(train) == 'False':
	train = False
else:
	train = True
load_model = args.load_model
if str(load_model) == 'False':
	load_model = False
else:
	load_model = True
# Environment setup
game = Game()

# ...
#Enter pygame environment here
# ...

torch.manual_seed(1); np.random.seed(1)
path = glob.glob(os.path.expanduser('./logs/'))[0]
SummaryWriter = SummaryWriter('{}{}'.format(path, datetime.now().strftime('%b%d_%H-%M-%S')))

# Print some info about the environment
print("State space (gym calls it observation space)")
#print(env.observation_space)
print("\nAction space")
#print(env.action_space.sample)

# Parameters
if train == False:
	epsilon = 0.0
else:
	epsilon = 0.2

discount_factor = 0.99
learning_rate = 0.01
n_successes = 0
max_position = -0.4

NUM_EPISODES = 1000
LEN_EPISODE = 500
reward_history = []

# initialize model
nn_model = model_network()
if load_model == True:
	nn_model.load_state_dict(torch.load('./models/checkpoint_final2.pth'), strict=False)
if train == False:
	nn_model.eval()
else :
	nn_model.train()
	print("training...")
	
# Performance metric
recent_reward=[]

loss_fn =  nn.MSELoss() # nn.SmoothL1Loss()
optimizer = optim.Adam(nn_model.parameters(), lr = learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 100, gamma = 0.9)

model_path = './models/checkpoint_final3.pth'

# Run for NUM_EPISODES
for episode in trange(NUM_EPISODES):
	episode_reward = 0
	episode_loss = 0
	# curr_state = env.reset()
	# curr_state object at the start of the episode from pygame
	curr_state = game.reset()
	# game.setTargetID(0)
	curr_state = np.asarray(curr_state)

	if train and episode%500 == 0:
		torch.save(nn_model.state_dict(), model_path)

	for step in range(LEN_EPISODE):
		# Comment to stop rendering the environment
		# If you don't render, you can speed things up
		# env.render()
		# game render here
		game.render()

		# get Q value for current state
		Q = nn_model(Variable(torch.from_numpy(curr_state).type(torch.FloatTensor)))

		# Randomly sample an action from the action space
		# Should really be your exploration/exploitation policy
		#action = env.action_space.sample()
		# Choosing epsilon-greedy action
		if train:
			if np.random.rand(1) < epsilon:
				action = np.random.randint(0,5)
			else:
				# take the action with maximum Q value
				_, action = torch.max(Q, -1)
				action = action.item()
			# print("hgf")
		else:
			# take the action with maximum Q value
			_, action = torch.max(Q, -1)
			action = action.item()
		# print(action)
		# Step forward and receive next state and reward
		# done flag is set when the episode ends: either goal is reached or
		#       200 steps are done
		#next_state, reward, done, _ = env.step(action)
		
		# ...
		# Execute 1 step in pygame
		# ...
		next_state, reward, gameOverFlag, gameSuccessFlag = game.step(action)
		next_state = np.asarray(next_state)
		
		# This is where your NN/GP code should go
		Q1 = nn_model(Variable(torch.from_numpy(next_state).type(torch.FloatTensor)))
		maxQ1, _ = torch.max(Q1,-1)
		
		# Create target vector
		Q_target = Q.clone()
		Q_target = Variable(Q_target.data)
		Q_target[action] = reward + torch.mul(maxQ1.detach(), discount_factor)


		# Update the policy
		loss = loss_fn(Q, Q_target)
		# print(loss)
		# input('a')
		if train:
			# Train the network/GP
			nn_model.zero_grad()
			loss.backward()
			optimizer.step()
		
		# Record history
		episode_reward += reward
		if train:
			episode_loss += loss.item()

		# Current state for next step
		curr_state = next_state

		# record max position 
		if next_state[0] > max_position:
			max_position = next_state[0]
			SummaryWriter.add_scalar('data/max_position', max_position, episode)

		if gameOverFlag or gameSuccessFlag:
			# print(episode_reward)
			# decrease epsilon value as the number of successful runs increase
			if gameSuccessFlag:
				if epsilon > 0.01:
					epsilon *= 0.5
					SummaryWriter.add_scalar('data/epsilon', epsilon, episode)

				# adjust learning rate as model converges
				if train:
					# if episode%200 == 0:
					scheduler.step()
					SummaryWriter.add_scalar('data/learning_rate',optimizer.param_groups[0]['lr'], episode)

				n_successes += 1
				SummaryWriter.add_scalar('data/cumulative_successes', n_successes, episode)
				SummaryWriter.add_scalar('data/success', 1, episode)
			else:
				SummaryWriter.add_scalar('data/success', 0, episode)
			# Record history
			reward_history.append(episode_reward)
			recent_reward.append(episode_reward)

			SummaryWriter.add_scalar('data/episode_reward', episode_reward, episode)
			SummaryWriter.add_scalar('data/episode_loss', episode_loss, episode)
			# if train:
			# SummaryWriter.add_scalar('data/position', next_state[0], episode)
			break

			# # You may want to plot periodically instead of after every episode
			# # Otherwise, things will slow down
			# fig = plt.figure(1)
			# plt.clf()
			# plt.xlim([0,NUM_EPISODES])
			# plt.plot(reward_history,'ro')
			# plt.xlabel('Episode')
			# plt.ylabel('Reward')
			# plt.title('Reward Per Episode')
			# plt.pause(0.01)
			# fig.canvas.draw()
if train:
	torch.save(nn_model.state_dict(), model_path)
SummaryWriter.close()
pygame.quit()
print('successful episodes: {:d} - {:.4f}%'.format(n_successes, n_successes*100/NUM_EPISODES))
# print('num_episodes with reward > -175: {:d} - {:.4f}%'.format(n_175, n_175*100/NUM_EPISODES))
# print('num_episodes with reward > -150: {:d} - {:.4f}%'.format(n_150, n_150*100/NUM_EPISODES))
# print('num_episodes with reward > -100: {:d} - {:.4f}%'.format(n_100, n_100*100/NUM_EPISODES))
