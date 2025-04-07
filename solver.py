import heapq

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