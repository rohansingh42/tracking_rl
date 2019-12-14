import pygame
import numpy as np

# define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

class Target(pygame.sprite.Sprite):
	def __init__(self, size=5, stepSize=5, startX=50, startY=50, id=0, color=RED, width=720, height=480):
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
		self.rect.centerx = startX
		self.rect.centery = startY

	def update(self):
		if self.id == 0:
			if self.rect.centerx <= 100 and self.rect.centery > 100:		# Go up
				self.rect.centery -= self.stepSize
				return
			elif self.rect.centery <= 100 and self.rect.centerx < (self.width-100):			# Go right
				self.rect.centerx += self.stepSize
				return
			elif self.rect.centerx >= (self.width-100) and self.rect.centery < (self.height-100):		# Go down
				self.rect.centery += self.stepSize
				return
			elif self.rect.centery >= (self.height-100) and self.rect.centerx > 100:		# Go left
				self.rect.centerx -= self.stepSize
				return
			else:
				self.rect.centerx += self.stepSize
				# self.rect.centery += self.stepSize
				return