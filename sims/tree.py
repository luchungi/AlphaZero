import torch
import numpy as np

class Node:
    def __init__(self, prior):
        self.visit_count = 0    # number of times node was visited
        self.value_sum = 0      # sum of value evaluations
        self.prior = prior      # prior probability of selecting this node based on policy network
        self.state = None       # game state at this node
        self.children = {}      # dictionary of child nodes keyed by action

    def update_state(self, state):
        """
        Update the state of the node.
        """
        self.state = state

    def value(self):
        """
        Calculate the average value of the node.
        """
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def expanded(self):
        """
        Check if the node has been expanded (i.e., has children).
        """
        return len(self.children) > 0

    def expand(self, legal_actions, priors):
        """
        Expand the node by adding child nodes for each possible action.
        """
        for action in legal_actions:
            if isinstance(action, torch.Tensor):
                action = action.item()
            self.children[action] = Node(priors[action])

    def action_probs(self, temp=1):
        """
        Returns dictionary of visit counts for each child node with the action as key.
        """
        if self.expanded():
            total_counts = sum(child.visit_count for child in self.children.values())
            action_probs = {action: (child.visit_count / total_counts) ** (1 / temp)
                            for action, child in self.children.items()}
            return action_probs
        else:
            return {}

    def best_action(self):
        """
        Returns action with the highest visit count among child nodes.
        """
        if self.expanded():
            return max(self.children.items(), key=lambda item: item[1].visit_count)[0]
        else:
            return None

    def __repr__(self):
        """
        Debugger pretty print node info
        """
        prior = "{0:.2f}".format(self.prior)
        return "{} Prior: {} Count: {} Value: {}".format(self.state.__str__(), prior, self.visit_count, self.value())

class MCTS:
    '''
    Monte Carlo Tree Search implementation for AlphaZero for two-player zero-sum games.
    The actions vector contains the probability of all possible actions in the game including illegal moves.
    The game function get_legal_actions returns the indices of legal moves in the current state.
    Reward is between -1, 0, 1, with 1 being a win for the current player, -1 a loss, and 0 a draw.
    Model should output a tuple of (action probabilities, value) given a state with value between -1 and 1.
    '''
    def __init__(self, game, model, args):
        self.game = game
        self.model = model # outputs action probabilities of all actions (including illegal ones)
        self.args = args # exploration constant for UCB calculation, higher values encourage exploration
        self.c_puct = args['c_puct']
        self.num_simulations = args['num_simulations']

    def select(self, node):
        """
        Select the child with the highest UCB score.
        """
        total_visits = sum(child.visit_count for child in node.children.values())
        best_score = -float('inf')
        best_action = None

        for action, child in node.children.items():
            q_value = -child.value()  # negative because we switch perspectives
            ucb_score = q_value + self.c_puct * child.prior * np.sqrt(total_visits) / (1 + child.visit_count)
            if ucb_score > best_score:
                best_score = ucb_score
                best_action = action

        return best_action.item() if isinstance(best_action, torch.Tensor) else best_action

    def run(self, state):
        root = Node(0)
        root.state = state
        self.expand_node(root, state)

        for i in range(self.num_simulations):
            node = root
            search_path = [node]

            # search through tree based on UCB scores until we reach a leaf node
            while node.expanded():
                action = self.select(node)
                node = node.children[action]
                search_path.append(node)

            # Expansion of leaf node
            parent_state = search_path[-2].state
            if node.state is None: # if state not yet set
                leaf_state = self.game.get_next_state(parent_state, action)
                node.update_state(leaf_state)
            else:
                leaf_state = node.state

            terminal_reward = self.game.check_winner(leaf_state) # returns None if not terminal, else reward for current player
            if terminal_reward is not None:
                # If the game has ended at this leaf node
                value = terminal_reward
            else:
                # If the game has not ended, use the model to get priors and value
                value = self.expand_node(node, leaf_state)

            # Backpropagation
            self.backpropagate(search_path, value)

        return root

    def expand_node(self, node, state):
        """
        Expand the node with legal actions and their prior probabilities.
        """
        with torch.no_grad():
            priors, value = self.model(state)
        legal_actions = self.game.get_legal_actions(state)
        node.expand(legal_actions, priors)
        return value

    @staticmethod
    def backpropagate(search_path, value):
        """
        Backpropagate the value up the search path.
        """
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += value
            value = -value  # switch perspective for the opponent assuming two-player zero-sum game