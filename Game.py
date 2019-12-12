"""
move4 - 4 dim action space - up(0), down(1), left(2), right(3), stay(4)

agent - GREEN
target - RED
obstacle - BLACK
"""

import pygame
import numpy as np

# define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

class Agent(pygame.sprite.Sprite):
	def __init__(self, size=5, startX=10, startY=10, color=GREEN,
			stepSize = 5, nLidarPoints = 4, lidarLength = 8):
		pygame.sprite.Sprite.__init__(self)
		self.size = stepSize
		self.color = color
		self.startX = startX
		self.startY = startY
		self.currX = startX
		self.currY = startY
		self.stepSize = stepSize
		self.lidarLength = lidarLength
		self.nLidarPoints = nLidarPoints

		## Create surface for agent and draw agent in it
		self.image = pygame.Surface((size,size))
		self.image.fill(color)
		self.rect = self.image.get_rect()
		self.image = self.image.convert()

	def move4(self, action, background, noObsFlag = True):
		backSize = background.get_size()
		gameOverFlag = False
		if noObsFlag == True:
			if action == 2:			# left
				if (self.currX-self.stepSize > 0):
					self.currX = self.currX - self.stepSize
				else:
					gameOverFlag = True

			if action == 3:			# right
				if (self.currX-self.stepSize < backSize[1]-1):
					self.currX = self.currX + self.stepSize
				else:
					gameOverFlag = True

			if action == 0:			# up
				if (self.currY-self.stepSize > 0):
					self.currY = self.currY - self.stepSize
				else:
					gameOverFlag = True

			if action == 1:			# down
				if (self.currY+self.stepSize < backSize[0]-1):
					self.currY = self.currY + self.stepSize
				else:
					gameOverFlag = True

		else:
			if action == 0:
				if (background.get_at((self.currX-self.stepSize,self.currY)) != (0,0,0)) and (self.currX-self.stepSize > 0):
					self.currX = self.currX - self.stepSize
				else:
					gameOverFlag = True

			if action == 1:
				if (background.get_at((self.currX+self.stepSize,self.currY)) != (0,0,0)) and (self.currX-self.stepSize < backSize[1]-1):
					self.currX = self.currX + self.stepSize
				else:
					gameOverFlag = True

			if action == 2:
				if (background.get_at((self.currX,self.currY-self.stepSize)) != (0,0,0)) and (self.currY-self.stepSize > 0):
					self.currY = self.currY - self.stepSize
				else:
					gameOverFlag = True

			if action == 3:
				if (background.get_at((self.currX,self.currY+self.stepSize)) != (0,0,0)) and (self.currY+self.stepSize < backSize[0]-1):
					self.currY = self.currY + self.stepSize
				else:
					gameOverFlag = True

		return self.getState(background), gameOverFlag
	
	def update(self):
		self.rect.x = self.currX
		self.rect.y = self.currY
			

	## 8-point detection map
	def visibility8(self, background):
		# like-clock
		visMap = [(self.currX - self.lidarLength, self.currY),		# u
			(self.currX - self.lidarLength, self.currY + self.lidarLength),		# u-r
			(self.currX, self.currY + self.lidarLength),		# r
			(self.currX + self.lidarLength, self.currY + self.lidarLength),		# d-r
			(self.currX + self.lidarLength, self.currY),		# d
			(self.currX + self.lidarLength, self.currY - self.lidarLength),		# d-l
			(self.currX, self.currY - self.lidarLength),		# l
			(self.currX - self.lidarLength, self.currY - self.lidarLength)]		# u-l

		# output = [-1 for x in visMap if background.get_at(x) == BLACK]
		# output = []
		output = [0, 0, 0, 0, 0, 0, 0, 0]
		for index, pt in enumerate(visMap):	
			if ((pt[0] > background.get_width()) or (pt[0] < 0)) or ((pt[1] > background.get_height()) or (pt[1] < 0)):
				output[index] = -1
			else:
				if (background.get_at(pt) == BLACK):
					output[index] = -1
				elif (background.get_at(pt) == RED):
					output[index]  = 1
		
		return output

	def getState(self, background):
		return [self.currX, self.currY] + self.visibility8(background)


	# ## 9-point lidar map
	# def visibility9(self, background):
	# 	# like-clock
	# 	visMap = [(self.currX - self.lidarLength, self.currY),		# u
	# 				(self.currX - self.lidarLength, self.currY + self.lidarLength),		# u-r
	# 				(self.currX, self.currY + self.lidarLength),		# r
	# 				(self.currX + self.lidarLength, self.currY + self.lidarLength),		# d-r
	# 				(self.currX + self.lidarLength, self.currY),		# d
	# 				(self.currX + self.lidarLength, self.currY - self.lidarLength),		# d-l
	# 				(self.currX, self.currY - self.lidarLength),		# l
	# 				(self.currX - self.lidarLength, self.currY - self.lidarLength),		# u-l
	# 				(self.currX, self.currY)]		# c


class Target(pygame.sprite.Sprite):
	def __init__(self, size=5, stepSize=5, startX=50, startY=50, id=0, color=RED, width=1080, height=720):
		pygame.sprite.Sprite.__init__(self)
		self.id = id
		self.color = color
		self.size = size
		self.stepSize = stepSize
		self.startX = startX
		self.startY = startY
		self.width = width
		self.height = height

		## Create surface for targer and draw in it
		self.image = pygame.Surface((size,size))
		self.image.fill(color)
		self.rect = self.image.get_rect()
		self.image = self.image.convert()

	def update(self):
		if self.id == 0:
			if self.rect.x <= 100 and self.rect.y > 100:		# Go up
				self.rect.y -= self.stepSize
			elif self.rect.y <= 100 and self.rect.x < (self.width-100):			# Go right
				self.rect.x += self.stepSize
			elif self.rect.x >= (self.width-100) and self.rect.y < (self.height-100):		# Go down
				self.rect.y += self.stepSize
			elif self.rect.y >= (self.height-100) and self.rect.x > 100:		# Go left
				self.rect.x -= self.stepSize


class Game:
	def __init__(self, width=1080, height=720, fps=30):
		# pygame.init()
		pygame.display.set_caption("Press ESC to quit")
		self.width = width
		self.height = height
		self.screen = pygame.display.set_mode((self.width, self.height), pygame.DOUBLEBUF)
		self.background = pygame.Surface(self.screen.get_size()).convert()
		self.background.fill(WHITE)
		self.clock = pygame.time.Clock()
		self.fps = fps

		# initialize agent sprites
		self.allAgents = pygame.sprite.Group()
		self.agent1 = Agent(startX=np.random.randint(50,self.width-50),
				startY=np.random.randint(50,self.height-50),
				nLidarPoints = 10, lidarLength = 50, size=50)
		self.allAgents.add(self.agent1);

		# initialize target sprites
		self.allTargets = pygame.sprite.Group()
		self.target1 = Target(startX=100, startY=100, id=0, size=50)
		self.allTargets.add(self.target1)

	def new(self):
		# initialize agent sprites
		self.allAgents = pygame.sprite.Group()
		self.agent1 = Agent(startX=np.random.randint(self.width),
				startY=np.random.randint(self.height), nLidarPoints = 10, lidarLength = 50)
		self.allAgents.add(self.agent1);

		# initialize target sprites
		self.allTargets = pygame.sprite.Group()
		self.target1 = Target(startX=100, startY=100, id=0)
		self.allTargets.add(self.target1)

	"""
	reward for 8 dim vector at each time step
	No target seen = -1 
	Target seen by only 1 sensor = 1
	Target seen by more than 1 sensor = 2
	"""

	def reward(self, state):
		count = 0
		for s in state:
			if s == 1:
				count += 1

		if count == 1:
			reward = 1
		elif  count > 1:
			reward = 2
		else:
			reward = -1
		return reward

	def step(self, action):
		self.allTargets.update()
		newState, gameOverFlag = self.agent1.move4(action=action, background=self.background)
		reward = self.reward(newState[2:])
		if gameOverFlag == True:
			reward = -100
		
		self.allAgents.update()

		return newState, reward, gameOverFlag

	def render(self):
		self.screen.blit(self.background,(0,0))
		self.allTargets.draw(self.screen)
		self.allAgents.draw(self.screen)
		pygame.display.flip()
