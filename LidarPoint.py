import pygame
import numpy as np

# define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

class LidarPoint(pygame.sprite.Sprite):
	def __init__(self, startX, startY, size=5, color=YELLOW, width=720, height=480):
		pygame.sprite.Sprite.__init__(self)
		self.color = color
		self.size = size
		self.startX = startX
		self.startY = startY
		self.currX = startX
		self.currY = startY
		self.width = width
		self.height = height

		## Create surface for targer and draw in it
		self.image = pygame.Surface((size,size))
		self.image.fill(color)
		self.rect = self.image.get_rect()
		self.image = self.image.convert()

	def plot(self, dx, dy):
		self.currX += dx
		self.currY += dy
		return [self.currX, self.currY]

	def wallCollide(self):
		if (self.currX>0 and self.currX<self.width) and (self.currY>0 and self.currY<self.height):
			return False
		else:
			return True

	def update(self):
		self.rect.centerx = self.currX
		self.rect.centery = self.currY