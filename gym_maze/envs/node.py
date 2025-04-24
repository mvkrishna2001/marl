
''' Adapted from https://github.com/rpinsler/gym-maze/tree/master'''

class Node(object):
    """Each Node in a search tree consists of:
            1. Current state
            2. A pointer to previous node
            3. The action leads to current state from previous one
            4. Path cost from the root to current state. 
    """
    
    def __init__(self, state, prev_node=None, action=None, step_cost=1):
        """Create a Node given current state, previous node, action and one-step cost"""
        self.state = state
        self.prev_node = prev_node
        self.action = action
        if self.prev_node is None:  # at root
            self.path_cost = 0
        else:
            self.path_cost = self.prev_node.path_cost + step_cost  # accumulate cost
            
    def __repr__(self):
        """String representation of the node."""
        return 'State: {}, Path cost: {}'.format(self.state, self.path_cost)
    
    def __lt__(self, node):
        """Operator of less than"""
        return self.path_cost < node.path_cost
    
    def next_node(self, maze, action):
        """Returns a Node with next state by taking an action from current state"""
        next_state = env._next_state(self.state, action)
        step_cost = maze.step_cost(self.state, action, next_state)
        
        return Node(next_state, self, action, step_cost)
