from gym_maze.envs.node import Node

from gym_maze.envs.algorithms.a_star import Frontier  # reuse Frontier class from A*

class DijkstraSolver:
    """Dijkstra's algorithm solver for the maze"""
    def __init__(self, env, goal):
        self.env = env
        self.goal = goal
        self.solution_node = self._dijkstra_search()

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

    def _dijkstra_search(self):
        f = lambda node: node.path_cost  # No heuristic
        frontier = Frontier(Node(self.env.state), f)
        explored = set()

        while frontier:
            node = frontier.pop()
            if tuple(node.state) == tuple(self.goal):
                return node

            explored.add(tuple(node.state))

            for action in self.env.all_actions:
                child_state = self.env._next_state(node.state, action)
                child = Node(child_state, node, action, step_cost=1)
                cstate = tuple(child.state)

                if cstate not in explored and cstate not in frontier:
                    frontier.add(child)
                elif cstate in frontier and frontier[cstate] < child:
                    frontier.replace(child)

        return None
