"""
move4 - 4 dim action space - up(0), down(1), left(2), right(3), stay(4)

agent - GREEN
target - RED
obstacle - BLACK
"""

import pygame
import numpy as np
from Agent import Agent
from Target import Target
from LidarPoint import LidarPoint

# define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

class Game:
	def __init__(self, width=720, height=480, fps=30, nLidarPoints = 10, lidarLength = 100, stepSize=20, agentSize=20):
		# pygame.init()
		pygame.display.set_caption("Press ESC to quit")
		self.width = width
		self.height = height
		self.screen = pygame.display.set_mode((self.width, self.height), pygame.DOUBLEBUF)
		self.background = pygame.Surface(self.screen.get_size()).convert()
		self.background.fill(WHITE)
		self.clock = pygame.time.Clock()
		self.fps = fps
		self.nLidarPoints = nLidarPoints
		self.lidarLength = lidarLength
		self.agentSize = agentSize
		self.stepSize = stepSize

		self.reset()
		
	def reset(self):
		# Number of iterations of game
		self.stepCount = 0

		# initialize agent sprites
		self.allAgents = pygame.sprite.Group()
		agentStartX = (np.random.randint(1,int(self.width/self.agentSize)-1)*self.agentSize)
		agentStartY = (np.random.randint(1,int(self.height/self.agentSize)-1)*self.agentSize)
		self.agent1 = Agent(startX=agentStartX, startY=agentStartY,
				nLidarPoints = 10, lidarLength = 80,
				size=self.agentSize, stepSize=self.stepSize)
		self.allAgents.add(self.agent1)

		# initialize target sprites
		self.allTargets = pygame.sprite.Group()
		targetStartX = (np.random.randint(1,int(self.width/self.agentSize)-1)*self.agentSize)
		targetStartY = (np.random.randint(1,int(self.height/self.agentSize)-1)*self.agentSize)
		self.target1 = Target(startX=100, startY=100, id=0,
				size=self.agentSize, stepSize=self.stepSize)
		self.allTargets.add(self.target1)

		# intialize lidar sprites
		dist = int(self.lidarLength/self.nLidarPoints)
		distDiag = int(self.lidarLength/(self.nLidarPoints*(2**0.5)))
		self.lidarUGroup = pygame.sprite.Group()
		self.lidarU = []
		for i in range(self.nLidarPoints):
			lp = LidarPoint(agentStartX, agentStartY - i*dist)
			self.lidarU.append(lp)
			self.lidarUGroup.add(lp)

		self.lidarURGroup = pygame.sprite.Group()
		self.lidarUR = []
		for i in range(self.nLidarPoints):
			lp = LidarPoint(agentStartX + i*distDiag, agentStartY - i*distDiag)
			self.lidarUR.append(lp)
			self.lidarURGroup.add(lp)

		self.lidarRGroup = pygame.sprite.Group()
		self.lidarR = []
		for i in range(self.nLidarPoints):
			lp = LidarPoint(agentStartX + i*dist, agentStartY)
			self.lidarR.append(lp)
			self.lidarRGroup.add(lp)

		self.lidarDRGroup = pygame.sprite.Group()
		self.lidarDR = []
		for i in range(self.nLidarPoints):
			lp = LidarPoint(agentStartX + i*distDiag, agentStartY + i*distDiag)
			self.lidarDR.append(lp)
			self.lidarDRGroup.add(lp)

		self.lidarDGroup = pygame.sprite.Group()
		self.lidarD = []
		for i in range(self.nLidarPoints):
			lp = LidarPoint(agentStartX, agentStartY + i*dist)
			self.lidarD.append(lp)
			self.lidarDGroup.add(lp)

		self.lidarDLGroup = pygame.sprite.Group()
		self.lidarDL = []
		for i in range(self.nLidarPoints):
			lp = LidarPoint(agentStartX - i*distDiag, agentStartY + i*distDiag)
			self.lidarDL.append(lp)
			self.lidarDLGroup.add(lp)

		self.lidarLGroup = pygame.sprite.Group()
		self.lidarL = []
		for i in range(self.nLidarPoints):
			lp = LidarPoint(agentStartX - i*dist, agentStartY)
			self.lidarL.append(lp)
			self.lidarLGroup.add(lp)

		self.lidarULGroup = pygame.sprite.Group()
		self.lidarUL = []
		for i in range(self.nLidarPoints):
			lp = LidarPoint(agentStartX - i*distDiag, agentStartY - i*distDiag)
			self.lidarUL.append(lp)
			self.lidarULGroup.add(lp)

		self.visibility8()

		return self.singleLidarOuput

	"""
	reward for 8 dim vector at each time step
	No target seen = -1
	Target seen by only 1 sensor = 1
	Target seen by more than 1 sensor = 2
	"""
	def reward(self):
		count = 0
		for s in self.singleLidarOuput:
			if s == 1:
				count += 1

		if count == 1:
			reward = 1
		elif  count > 1:
			reward = 2
		else:
			reward = -1
		return reward

	def visibility8(self):
		self.completeLidarOut = []
		self.singleLidarOuput = [0,0,0,0,0,0,0,0]
		
		temp = []
		for sp in self.lidarU:
			if sp.wallCollide():
				temp.append(-1)
				if self.singleLidarOuput[0] == 0:
					self.singleLidarOuput[0] = -1
			elif pygame.sprite.spritecollide(sp, self.allTargets, False):
				temp.append(1)
				self.singleLidarOuput[0] = 1 
			else:
				temp.append(0)
		self.completeLidarOut.append(temp)

		temp = []
		for sp in self.lidarUR:
			if sp.wallCollide():
				temp.append(-1)
				if self.singleLidarOuput[1] == 0:
					self.singleLidarOuput[1] = -1
			elif pygame.sprite.spritecollide(sp, self.allTargets, False):
				temp.append(1)
				self.singleLidarOuput[1] = 1 
			else:
				temp.append(0)
		self.completeLidarOut.append(temp)

		temp = []
		for sp in self.lidarR:
			if sp.wallCollide():
				temp.append(-1)
				if self.singleLidarOuput[2] == 0:
					self.singleLidarOuput[2] = -1
			elif pygame.sprite.spritecollide(sp, self.allTargets, False):
				temp.append(1)
				self.singleLidarOuput[2] = 1 
			else:
				temp.append(0)
		self.completeLidarOut.append(temp)

		temp = []
		for sp in self.lidarDR:
			if sp.wallCollide():
				temp.append(-1)
				if self.singleLidarOuput[3] == 0:
					self.singleLidarOuput[3] = -1
			elif pygame.sprite.spritecollide(sp, self.allTargets, False):
				temp.append(1)
				self.singleLidarOuput[3] = 1 
			else:
				temp.append(0)
		self.completeLidarOut.append(temp)

		temp = []
		for sp in self.lidarD:
			if sp.wallCollide():
				temp.append(-1)
				if self.singleLidarOuput[4] == 0:
					self.singleLidarOuput[4] = -1
			elif pygame.sprite.spritecollide(sp, self.allTargets, False):
				temp.append(1)
				self.singleLidarOuput[4] = 1 
			else:
				temp.append(0)
		self.completeLidarOut.append(temp)

		temp = []
		for sp in self.lidarDL:
			if sp.wallCollide():
				temp.append(-1)
				if self.singleLidarOuput[5] == 0:
					self.singleLidarOuput[5] = -1
			elif pygame.sprite.spritecollide(sp, self.allTargets, False):
				temp.append(1)
				self.singleLidarOuput[5] = 1 
			else:
				temp.append(0)
		self.completeLidarOut.append(temp)

		temp = []
		for sp in self.lidarL:
			if sp.wallCollide():
				temp.append(-1)
				if self.singleLidarOuput[6] == 0:
					self.singleLidarOuput[6] = -1
			elif pygame.sprite.spritecollide(sp, self.allTargets, False):
				temp.append(1)
				self.singleLidarOuput[6] = 1 
			else:
				temp.append(0)
		self.completeLidarOut.append(temp)

		temp = []
		for sp in self.lidarUL:
			if sp.wallCollide():
				temp.append(-1)
				if self.singleLidarOuput[7] == 0:
					self.singleLidarOuput[7] = -1
			elif pygame.sprite.spritecollide(sp, self.allTargets, False):
				temp.append(1)
				self.singleLidarOuput[7] = 1 
			else:
				temp.append(0)
		self.completeLidarOut.append(temp)

	def updateLidar(self, dx, dy):
		for sp in self.lidarU:
			sp.plot(dx,dy)
		self.lidarUGroup.update()
		for sp in self.lidarUR:
			sp.plot(dx,dy)
		self.lidarURGroup.update()
		for sp in self.lidarR:
			sp.plot(dx,dy)
		self.lidarRGroup.update()
		for sp in self.lidarDR:
			sp.plot(dx,dy)
		self.lidarDRGroup.update()
		for sp in self.lidarD:
			sp.plot(dx,dy)
		self.lidarDGroup.update()
		for sp in self.lidarDL:
			sp.plot(dx,dy)
		self.lidarDLGroup.update()
		for sp in self.lidarL:
			sp.plot(dx,dy)
		self.lidarLGroup.update()
		for sp in self.lidarUL:
			sp.plot(dx,dy)
		self.lidarULGroup.update()
	
	def step(self, action):

		gameSuccessFlag = False
		# calls need to follow this order
		self.allTargets.update()
		prevState = self.agent1.getState()
		newState, gameOverFlag = self.agent1.move4(action=action, background=self.background)
		self.updateLidar(newState[0]-prevState[0], newState[1]-prevState[1])
		self.visibility8()
		reward = self.reward()

		if reward >= 0:
			self.stepCount += 1

		# print(self.stepCount)
		if gameOverFlag == True:
<<<<<<< HEAD
			reward = -100
		elif self.stepCount > 200:
=======
			reward = -200
		elif self.stepCount > 100:
>>>>>>> 5c086f33b20689db4350ce67b40ef6aa0b478131
			gameSuccessFlag = True
			reward = 200
		
		# if self.stepCount > 500:
		# 	gameOverFlag = True
		# 	reward = -1
			
		self.allAgents.update()

		return self.singleLidarOuput, reward, gameOverFlag, gameSuccessFlag

	# def stepImage(self, action):
	# 	# calls need to follow this order
	# 	self.allTargets.update()
	# 	prevState = self.agent1.getState()
	# 	newState, gameOverFlag = self.agent1.move4(action=action, background=self.background)
	# 	self.updateLidar(newState[0]-prevState[0], newState[1]-prevState[1])
	# 	self.visibility8()
	# 	reward = self.reward()

	# 	if gameOverFlag == True:
	# 		reward = -100
	# 	self.allAgents.update()

	# 	# for image

	# 	self.screen.blit(self.background,(0,0))
	# 	self.allTargets.draw(self.screen)
	# 	self.allAgents.draw(self.screen)


		return self.singleLidarOuput, reward, gameOverFlag

	def render(self):
		self.screen.blit(self.background,(0,0))
		self.allTargets.draw(self.screen)
		self.allAgents.draw(self.screen)
		self.lidarUGroup.draw(self.screen)
		self.lidarURGroup.draw(self.screen)
		self.lidarRGroup.draw(self.screen)
		self.lidarDRGroup.draw(self.screen)
		self.lidarDGroup.draw(self.screen)
		self.lidarDLGroup.draw(self.screen)
		self.lidarLGroup.draw(self.screen)
		self.lidarULGroup.draw(self.screen)
		pygame.time.delay(10)
		pygame.display.flip()
