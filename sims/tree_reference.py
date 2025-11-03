import torch
import math
import numpy as np

class Node:
    def __init__(self, prior, to_play):
        self.visit_count = 0
        self.to_play = to_play # which player is to play at this node
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.state = None # only populated for expanded nodes

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def select_action(self, temperature):
        """
        Select action according to the visit count distribution and the temperature.
        """
        visit_counts = np.array([child.visit_count for child in self.children.values()])
        actions = [action for action in self.children.keys()]
        if temperature == 0:
            action = actions[np.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = np.random.choice(actions)
        else:
            # See paper appendix Data Generation
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(visit_count_distribution)
            action = np.random.choice(actions, p=visit_count_distribution)

        return action

    def __repr__(self):
        """
        Debugger pretty print node info
        """
        prior = "{0:.2f}".format(self.prior)
        return "{} Prior: {} Count: {} Value: {}".format(self.state.__str__(), prior, self.visit_count, self.value())


class MCTS:

    def __init__(self, game, model, args):
        self.game = game
        self.model = model
        self.args = args
        self.c_puct = args['c_puct']
        self.num_simulations = args['num_simulations']

    def run(self, state, to_play):

        root = Node(0, to_play)

        # EXPAND root
        action_probs, _ = self.get_prob_and_value(state)
        self.expand(root, state, to_play, action_probs)

        for _ in range(self.num_simulations):
            node = root
            search_path = [node]

            # SELECT
            while node.expanded():
                action, node = self.select_child(node)
                search_path.append(node)

            parent = search_path[-2]
            state = parent.state
            # Now we're at a leaf node and we would like to expand
            # Players always play from their own perspective
            next_state, _ = self.game.get_next_state(state, player=1, action=action)
            # Get the board from the perspective of the other player
            next_state = self.game.get_canonical_board(next_state, player=-1)

            # The value of the new state from the perspective of the other player
            value = self.game.get_reward_for_player(next_state, player=1)
            if value is None:
                # If the game has not ended, expand the node
                action_probs, value = self.get_prob_and_value(next_state)
                self.expand(node, next_state, parent.to_play * -1, action_probs)

            self.backpropagate(search_path, value, parent.to_play * -1)

        return root

    def select_child(self, node):
        """
        Select the child with the highest UCB score.
        """
        best_score = -np.inf
        best_action = -1
        best_child = None

        for action, child in node.children.items():
            score = self.ucb_score(node, child)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def expand(self, node, state, to_play, action_probs):
        """
        We expand a node and keep track of the prior policy probability given by neural network
        """
        node.to_play = to_play
        node.state = state
        for a, prob in enumerate(action_probs):
            if prob != 0:
                node.children[a] = Node(prior=prob, to_play=node.to_play * -1)

    def backpropagate(self, search_path, value, to_play):
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree
        to the root.
        """
        for node in reversed(search_path):
            node.value_sum += value if node.to_play == to_play else -value
            node.visit_count += 1

    def get_prob_and_value(self, state):
        with torch.no_grad():
            action_probs, value = self.model.predict(state)
        valid_moves = self.game.get_valid_moves(state)
        action_probs = action_probs * valid_moves  # mask invalid moves
        action_probs /= np.sum(action_probs)
        return action_probs, value


    def ucb_score(self, parent, child):
        '''
        Score based on Upper Confidence Bound = Q(s,a) + U(s,a)
        where Q(s,a) is the value of the child node, and U(s,a) is an exploration term
        U(s,a) = c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        '''
        U = self.c_puct * child.prior * math.sqrt(parent.visit_count) / (child.visit_count + 1)
        if child.visit_count > 0:
            # The value of the child is from the perspective of the opposing player
            Q = -child.value()
        else:
            Q = 0

        return Q + U


