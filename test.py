import pygame
import numpy as np

from Game import Game, Agent, Target

# pygame.init()

game = Game()

running = True
gof = False
ns = []
reward = 0
clock = pygame.time.Clock()

while running:
    # game.render()
    clock.tick(30)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                ns, reward, gof = game.step(0)
            elif event.key == pygame.K_DOWN:
                ns, reward, gof = game.step(1)
            elif event.key == pygame.K_LEFT:
                ns, reward, gof = game.step(2)
            elif event.key == pygame.K_RIGHT:
                ns, reward, gof = game.step(3)
        else:
            ns, reward, gof = game.step(4)

        if gof:
            running = False
    print("Reward : ",reward, "State : ", ns)

