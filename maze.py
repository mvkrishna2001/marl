import pygame
import random
import heapq

# Screen settings
WIDTH, HEIGHT = 600, 600
CELL_SIZE = 20
ROWS, COLS = HEIGHT // CELL_SIZE, WIDTH // CELL_SIZE

# Colors
WHITE, BLACK, GREEN, RED, BLUE = (255, 255, 255), (0, 0, 0), (0, 255, 0), (255, 0, 0), (0, 0, 255)

pygame.init()
# pygame.event.set_allowed([pygame.QUIT, pygame.KEYDOWN])
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# Maze generation using Recursive Backtracking
def generate_maze(rows, cols):
    maze = [[1] * cols for _ in range(rows)]
    stack = [(0, 0)]
    visited = set(stack)

    def neighbors(x, y):
        options = [(x+dx, y+dy) for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]]
        return [(nx, ny) for nx, ny in options if 0 <= nx < cols and 0 <= ny < rows and (nx, ny) not in visited]

    while stack:
        x, y = stack[-1]
        maze[y][x] = 0
        nb = neighbors(x, y)
        if nb:
            nx, ny = random.choice(nb)
            maze[(y+ny)//2][(x+nx)//2] = 0  # Remove wall
            visited.add((nx, ny))
            stack.append((nx, ny))
        else:
            stack.pop()

    return maze

maze = generate_maze(ROWS, COLS)

# A* Algorithm for Pathfinding
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan distance

def a_star(start, goal):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]  # Reverse path

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < COLS and 0 <= neighbor[1] < ROWS and maze[neighbor[1]][neighbor[0]] == 0:
                temp_g_score = g_score[current] + 1
                if neighbor not in g_score or temp_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = temp_g_score
                    f_score[neighbor] = temp_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return []  # No path found

def find_nearest_open_space(maze, x, y):
    """Finds the nearest open space (0) to the given (x, y) position."""
    from collections import deque

    queue = deque([(x, y)])
    visited = set(queue)

    while queue:
        cx, cy = queue.popleft()
        if maze[cy][cx] == 0:  # Found an open space
            return (cx, cy)

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Left, Right, Up, Down
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < len(maze[0]) and 0 <= ny < len(maze) and (nx, ny) not in visited:
                queue.append((nx, ny))
                visited.add((nx, ny))

    return None  # Should never happen in a valid maze


# Ensure start and goal are in open spaces
start_pos = find_nearest_open_space(maze, 0, 0)
goal_pos = find_nearest_open_space(maze, COLS - 1, ROWS - 1)

running = True

visited_positions = set()  # Store visited cells
solution_path = []  # Ensure solution path is initialized
solving = False
path_index = 0

while running:
    screen.fill(WHITE)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE and not solving: #To start solving, press the SPACE keybar
            # Only start solving if it hasn't started already
            solution_path = a_star(start_pos, goal_pos)  # Calculate path
            if solution_path:  # Ensure there's a path found
                solving = True
                path_index = 0  # Reset index
                visited_positions.clear()  # Reset visited path

    # Draw maze
    for y in range(ROWS):
        for x in range(COLS):
            if maze[y][x] == 1:
                pygame.draw.rect(screen, BLACK, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    # Draw start and goal
    pygame.draw.rect(screen, GREEN, (start_pos[0] * CELL_SIZE, start_pos[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    pygame.draw.rect(screen, RED, (goal_pos[0] * CELL_SIZE, goal_pos[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    # Solve maze step by step after pressing SPACE
    if solving and path_index < len(solution_path):
        x, y = solution_path[path_index]
        visited_positions.add((x, y))  # Store visited positions
        path_index += 1  # Reveal next step

    # Draw visited path (lighter blue)
    for x, y in visited_positions:
        pygame.draw.rect(screen, (100, 100, 255), (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))  # Light blue

    # Draw current position (darker blue)
    if solving and path_index < len(solution_path):
        x, y = solution_path[path_index]
        pygame.draw.rect(screen, BLUE, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    pygame.display.flip()
    clock.tick(10)  # Adjust speed of solving animation

pygame.quit()

