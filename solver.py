import heapq
from collections import deque

def a_star(start, goal, maze):
    rows, cols = len(maze), len(maze[0])
    open_set = [(0 + heuristic(start, goal), 0, start, [])]
    visited = set()

    while open_set:
        _, cost, current, path = heapq.heappop(open_set)
        if current in visited:
            continue
        path = path + [current]
        if current == goal:
            return path
        visited.add(current)
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx, ny = current[0] + dx, current[1] + dy
            if 0 <= nx < cols and 0 <= ny < rows and maze[ny][nx] == 0:
                heapq.heappush(open_set, (cost + 1 + heuristic((nx, ny), goal), cost + 1, (nx, ny), path))
    return []

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def bfs(start, goal, maze):
    rows, cols = len(maze), len(maze[0])
    queue = deque([(start, [])])  # (position, path)
    visited = set()
    visited.add(start)

    while queue:
        current, path = queue.popleft()
        if current == goal:
            return path + [current]

        x, y = current
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < cols and 0 <= ny < rows and maze[ny][nx] == 0 and (nx, ny) not in visited:
                queue.append(((nx, ny), path + [current]))
                visited.add((nx, ny))

    return []


def dfs(start, goal, maze):
    rows, cols = len(maze), len(maze[0])
    stack = [(start, [])]  # (position, path)
    visited = set()
    visited.add(start)

    while stack:
        current, path = stack.pop()
        if current == goal:
            return path + [current]

        x, y = current
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < cols and 0 <= ny < rows and maze[ny][nx] == 0 and (nx, ny) not in visited:
                stack.append(((nx, ny), path + [current]))
                visited.add((nx, ny))

    return []

def dijkstra(start, goal, maze):
    rows, cols = len(maze), len(maze[0])
    queue = [(0, start, [])]  # (distance, position, path)
    distances = {start: 0}
    visited = set()

    while queue:
        current_dist, current, path = heapq.heappop(queue)

        if current == goal:
            return path + [current]

        if current in visited:
            continue
        visited.add(current)

        x, y = current
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < cols and 0 <= ny < rows and maze[ny][nx] == 0:
                new_dist = current_dist + 1
                if (nx, ny) not in distances or new_dist < distances[(nx, ny)]:
                    distances[(nx, ny)] = new_dist
                    heapq.heappush(queue, (new_dist, (nx, ny), path + [current]))

    return []