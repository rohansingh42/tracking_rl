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
YELLOW = (255, 255, 0)

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
				if (self.currX+self.stepSize < backSize[0]-1):
					self.currX = self.currX + self.stepSize
				else:
					gameOverFlag = True

			if action == 0:			# up
				if (self.currY-self.stepSize > 0):
					self.currY = self.currY - self.stepSize
				else:
					gameOverFlag = True

			if action == 1:			# down
				if (self.currY+self.stepSize < backSize[1]-1):
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

		return self.getState(), gameOverFlag

	def update(self):
		self.rect.centerx = self.currX
		self.rect.centery = self.currY


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

	def getState(self):
		return [self.currX, self.currY]# + self.visibility8(background)


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