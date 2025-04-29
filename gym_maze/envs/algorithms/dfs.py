from gym_maze.envs.node import Node

class DFSSolver:
    """Depth-First Search solver for the maze"""
    def __init__(self, env, goal, start_state):
        self.env = env
        self.goal = goal
        self.start_state = start_state
        self.solution_node = self._dfs_search()

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

    def _dfs_search(self):
        start_node = Node(self.start_state)
        stack = [start_node]
        explored = set()

        while stack:
            node = stack.pop()
            if tuple(node.state) == tuple(self.goal):
                return node

            explored.add(tuple(node.state))

            for action in self.env.all_actions:
                next_state = self.env._next_state(node.state, action)
                if tuple(next_state) not in explored:
                    child = Node(next_state, node, action)
                    stack.append(child)

        return None  # No solution found