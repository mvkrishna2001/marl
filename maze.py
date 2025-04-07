import random

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
