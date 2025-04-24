from collections import deque
from gym_maze.envs.node import Node

class BFSSolver(object):
    """Breadth-First Search solver for the maze."""
    def __init__(self, env, goal):
        self.env = env
        self.goal = goal
        self.solution_node = self._bfs_search()

    def solvable(self):
        return self.solution_node is not None

    def get_actions(self):
        node = self.solution_node
        actions = []
        while node.prev_node:
            actions.append(node.action)
            node = node.prev_node
        return actions[::-1]

    def get_states(self):
        node = self.solution_node
        states = [node.state]
        while node.prev_node:
            node = node.prev_node
            states.append(node.state)
        return states[::-1]

    def _bfs_search(self):
        frontier = deque([Node(self.env.state)])
        explored = set()

        while frontier:
            node = frontier.popleft()
            if tuple(node.state) == tuple(self.goal):
                return node
            explored.add(tuple(node.state))

            for action in self.env.all_actions:
                child_state = self.env._next_state(node.state, action)
                child = Node(child_state, node, action)
                child_key = tuple(child.state)

                if child_key not in explored and all(child_key != tuple(n.state) for n in frontier):
                    frontier.append(child)

        return None
