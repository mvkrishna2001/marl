import pygame
import argparse

from config import *
from maze import generate_maze, find_nearest_open_space
from agent import Agent

pygame.init()
screen = pygame.display.set_mode((COLS * CELL_SIZE, ROWS * CELL_SIZE))
clock = pygame.time.Clock()
running = True
solving = False  # ← flag to control solving start

maze = generate_maze(ROWS, COLS)

# Command-line argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Choose the maze solver algorithm.")
    parser.add_argument('--algorithm', type=str, choices=['a_star', 'bfs', 'dfs', 'dijkstra'], default='a_star',
                        help="The algorithm to use for solving the maze.")
    return parser.parse_args()

args = parse_args()

# Define agent configs with start/goal colors
# config: name, start, goal, color, maze, start_color, goal_color
agent_configs = [
    ("A", (0, 0), (COLS-1, ROWS-1), (0, 0, 255), (0, 255, 255), (255, 0, 255), args.algorithm),
    # ("B", (0, ROWS-1), (COLS-1, 0), (255, 0, 0), (255, 255, 0), (255, 100, 100)),
    # ("C", (COLS//2, 0), (COLS//2, ROWS-1), (0, 255, 0), (0, 100, 0), (100, 255, 100)),
]

agents = []
for name, s_coord, g_coord, color, start_color, goal_color, algorithm in agent_configs:
    start = find_nearest_open_space(maze, *s_coord)
    goal = find_nearest_open_space(maze, *g_coord)
    agents.append(Agent(name, start, goal, color, maze, start_color, goal_color))

while running:
    screen.fill(WHITE)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            solving = True

    # Draw maze
    for y in range(ROWS):
        for x in range(COLS):
            if maze[y][x] == 1:
                pygame.draw.rect(screen, BLACK, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    # Solve and draw agents
    for agent in agents:
        if solving:
            agent.move()
        agent.draw(screen)

    pygame.display.flip()
    clock.tick(10)

pygame.quit()
