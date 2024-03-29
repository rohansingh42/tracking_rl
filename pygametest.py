#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pygame as pg

pg.init()

screen = pg.display.set_mode((1080, 720))
background = pg.Surface(screen.get_size())
# screen.fill((255,255,255))
background.fill((255,255,255))
# background = background.convert()

screen.blit(background, (0,0))
print(background.get_at((20,20)))
pg.display.flip()
test = pg.Surface((10,10))
test.fill((255,0,0))
background.blit(test,(20,20))
print(background.get_at((20,20)))
screen.blit(background, (0,0))
pg.display.flip()




mainloop = True

while mainloop:
	for event in pg.event.get():
	    if event.type == pg.QUIT: 
	        mainloop = False # pygame window closed by user
	    elif event.type == pg.KEYDOWN:
	        if event.key == pg.K_ESCAPE:
	            mainloop = False # user pressed ESC

	# screen.blit(background, (0,0))

# """
# 002_display_fps_pretty.py

# Display framerate and playtime.
# Works with Python 2.7 and 3.3+.

# URL:     http://thepythongamebook.com/en:part2:pygame:step002
# Author:  yipyip
# License: Do What The Fuck You Want To Public License (WTFPL)
#          See http://sam.zoy.org/wtfpl/
# """

# ####

# import pygame


# ####

# class PygView(object):


#     def __init__(self, width=640, height=400, fps=30):
#         """Initialize pygame, window, background, font,...
#         """
#         pygame.init()
#         pygame.display.set_caption("Press ESC to quit")
#         self.width = width
#         self.height = height
#         #self.height = width // 4
#         self.screen = pygame.display.set_mode((self.width, self.height), pygame.DOUBLEBUF)
#         self.background = pygame.Surface(self.screen.get_size()).convert()
#         self.clock = pygame.time.Clock()
#         self.fps = fps
#         self.playtime = 0.0
#         self.font = pygame.font.SysFont('mono', 20, bold=True)


#     def run(self):
#         """The mainloop
#         """
#         running = True
#         while running:
#             for event in pygame.event.get():
#                 if event.type == pygame.QUIT:
#                     running = False
#                 elif event.type == pygame.KEYDOWN:
#                     if event.key == pygame.K_ESCAPE:
#                         running = False

#             milliseconds = self.clock.tick(self.fps)
#             self.playtime += milliseconds / 1000.0
#             self.draw_text("FPS: {:6.3}{}PLAYTIME: {:6.3} SECONDS".format(
#                            self.clock.get_fps(), " "*5, self.playtime))

#             pygame.display.flip()
#             self.screen.blit(self.background, (0, 0))

#         pygame.quit()


#     def draw_text(self, text):
#         """Center text in window
#         """
#         fw, fh = self.font.size(text) # fw: font width,  fh: font height
#         surface = self.font.render(text, True, (0, 255, 0))
#         # // makes integer division in python3
#         self.screen.blit(surface, ((self.width - fw) // 2, (self.height - fh) // 2))

# ####

# if __name__ == '__main__':

#     # call with width of window and fps
#     PygView(640, 400).run()