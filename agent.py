import pygame

from solver import *
from config import CELL_SIZE

class Agent:
    def __init__(self, name, start, goal, color, maze, start_color, goal_color, algorithm='a_star'):
        self.name = name
        self.start = start
        self.goal = goal
        self.color = color
        self.algorithm = algorithm
        self.start_color = start_color
        self.goal_color = goal_color
        self.path = self.choose_algorithm(start, goal, maze)
        self.path_index = 0
        self.visited = set()

    def choose_algorithm(self, start, goal, maze):
        if self.algorithm == 'bfs':
            return bfs(start, goal, maze)
        elif self.algorithm == 'dfs':
            return dfs(start, goal, maze)
        elif self.algorithm == 'dijkstra':
            return dijkstra(start, goal, maze)
        else: #default algorithm is A*
            return a_star(start, goal, maze)


    def move(self):
        if self.path_index < len(self.path):
            pos = self.path[self.path_index]
            self.visited.add(pos)
            self.path_index += 1
            return pos
        return None

    def draw(self, screen):
        # Trail
        for x, y in self.visited:
            pygame.draw.rect(screen, self.color, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        # Current position
        if self.path_index < len(self.path):
            x, y = self.path[self.path_index]
            highlight = tuple(min(c+50,255) for c in self.color)
            pygame.draw.rect(screen, highlight, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        # Start and goal
        sx, sy = self.start
        gx, gy = self.goal
        pygame.draw.rect(screen, self.start_color, (sx * CELL_SIZE, sy * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        pygame.draw.rect(screen, self.goal_color, (gx * CELL_SIZE, gy * CELL_SIZE, CELL_SIZE, CELL_SIZE))
