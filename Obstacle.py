import pygame
import numpy as np

# define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

class Obstacle(pygame.sprite.Sprite):
	def __init__(self, obWidth, obHeight, startX, startY, color=BLACK, width=720, height=480):
		pygame.sprite.Sprite.__init__(self)
		self.color = color
		self.size = size
		self.startX = startX
		self.startY = startY
		self.currX = startX
		self.currY = startY
		self.width = width
		self.height = height
		self.obHeight = obHeight
		self.obWidth = obWidth

		## Create surface for targer and draw in it
		self.image = pygame.Surface((self.obWidth, self.obHeight))
		self.image.fill(color)
		self.rect = self.image.get_rect()
		self.image = self.image.convert()
		self.rect.x = startX
		self.rect.y = startY

	def getPos(self):
		return [self.currX, self.currY, self.obWidth, self.obHeight]

	def update(self):
		return