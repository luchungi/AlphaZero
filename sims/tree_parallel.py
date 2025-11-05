import torch
import numpy as np
from collections import defaultdict, deque


class Node:
    """Node in the MCTS tree."""

    def __init__(self, prior):
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior
        self.state = None
        self.children = {}

    def update_state(self, state):
        self.state = state

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def expanded(self):
        return len(self.children) > 0

    def expand(self, legal_actions, priors):
        for action in legal_actions:
            if isinstance(action, torch.Tensor):
                action = action.item()
            self.children[action] = Node(priors[action].item() if isinstance(priors[action], torch.Tensor) else priors[action])

    def action_probs(self, temp=1):
        if self.expanded():
            total_counts = sum(child.visit_count for child in self.children.values())
            if total_counts == 0:
                return {}
            action_probs = {action: (child.visit_count / total_counts) ** (1 / temp)
                            for action, child in self.children.items()}
            # Normalize
            total = sum(action_probs.values())
            if total > 0:
                action_probs = {k: v / total for k, v in action_probs.items()}
            return action_probs
        return {}

    def best_action(self):
        if self.expanded():
            return max(self.children.items(), key=lambda item: item[1].visit_count)[0]
        return None

    def sample_action(self):
        if self.expanded():
            visit_counts = np.array([child.visit_count for child in self.children.values()])
            actions = list(self.children.keys())
            if visit_counts.sum() == 0:
                return np.random.choice(actions)
            probabilities = visit_counts / visit_counts.sum()
            return np.random.choice(actions, p=probabilities)
        return None


class ParallelMCTS:
    """
    Parallel MCTS implementation with batched model predictions.

    This implementation batches neural network predictions across multiple
    MCTS simulations, significantly improving throughput when using GPUs.
    """

    def __init__(self, game, model, args):
        """
        Args:
            game: Game interface with get_next_state, get_legal_actions, reward methods
            model: Batched model wrapper for predictions
            args: Dictionary with 'c_puct', 'num_simulations', 'batch_size'
        """
        self.game = game
        self.model = model
        self.args = args
        self.c_puct = args.get('c_puct', 1.0)
        self.num_simulations = args.get('num_simulations', 800)
        self.batch_size = args.get('batch_size', 16)

    def select(self, node):
        """Select child with highest UCB score."""
        total_visits = sum(child.visit_count for child in node.children.values())
        best_score = -float('inf')
        best_action = None

        for action, child in node.children.items():
            q_value = -child.value()
            ucb_score = q_value + self.c_puct * child.prior * np.sqrt(total_visits) / (1 + child.visit_count)
            if ucb_score > best_score:
                best_score = ucb_score
                best_action = action

        return best_action

    def run(self, state):
        """
        Run MCTS with batched model predictions.

        Args:
            state: Initial game state

        Returns:
            root: Root node of the search tree
        """
        root = Node(0)
        root.state = state.clone()

        # Initial expansion of root
        terminal_reward = self.game.reward(root.state)
        if terminal_reward is None:
            self._expand_node_single(root, root.state)
        else:
            return root

        # Batch simulations for efficiency
        num_batches = (self.num_simulations + self.batch_size - 1) // self.batch_size

        for batch_idx in range(num_batches):
            batch_size = min(self.batch_size, self.num_simulations - batch_idx * self.batch_size)
            self._run_batch_simulations(root, batch_size)

        return root

    def _run_batch_simulations(self, root, batch_size):
        """Run a batch of MCTS simulations with batched model predictions."""
        # Data structures for batch
        search_paths = []
        leaf_nodes = []
        leaf_states = []
        leaf_actions = []

        # Selection phase - traverse tree to leaves
        for _ in range(batch_size):
            node = root
            search_path = [node]

            # Navigate to leaf
            while node.expanded():
                action = self.select(node)
                node = node.children[action]
                search_path.append(node)

            search_paths.append(search_path)
            leaf_nodes.append(node)

            # Get leaf state
            if node.state is None:
                parent_state = search_path[-2].state
                action = [a for a, child in search_path[-2].children.items() if child == node][0]
                leaf_state = self.game.get_next_state(parent_state, action)
                node.update_state(leaf_state)
                leaf_actions.append(action)
            else:
                leaf_state = node.state

            leaf_states.append(leaf_state)

        # Check for terminal states
        terminal_values = []
        non_terminal_indices = []
        non_terminal_states = []
        non_terminal_nodes = []

        for i, state in enumerate(leaf_states):
            terminal_reward = self.game.reward(state)
            if terminal_reward is not None:
                terminal_values.append((i, terminal_reward))
            else:
                non_terminal_indices.append(i)
                non_terminal_states.append(state)
                non_terminal_nodes.append(leaf_nodes[i])

        # Batched model prediction for non-terminal states
        values = [None] * batch_size
        if non_terminal_states:
            states_tensor = torch.stack(non_terminal_states)
            policies, pred_values = self.model.predict_batch(states_tensor)

            # Expand nodes
            for i, node_idx in enumerate(non_terminal_indices):
                node = non_terminal_nodes[i]
                state = non_terminal_states[i]
                policy = policies[i]
                value = pred_values[i].item()

                legal_actions = self.game.get_legal_actions(state)
                node.expand(legal_actions, policy.cpu())
                values[node_idx] = value

        # Fill in terminal values
        for i, terminal_value in terminal_values:
            values[i] = terminal_value

        # Backpropagation
        for search_path, value in zip(search_paths, values):
            if value is not None:
                self._backpropagate(search_path, value)

    def _expand_node_single(self, node, state):
        """Expand a single node (used for root initialization)."""
        policy, value = self.model.predict_single(state)
        legal_actions = self.game.get_legal_actions(state)
        node.expand(legal_actions, policy.cpu())
        return value.item()

    @staticmethod
    def _backpropagate(search_path, value):
        """Backpropagate value up the search path."""
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += value
            value = -value


class VirtualLossMCTS:
    """
    MCTS with virtual loss for parallel tree traversal.

    Virtual loss temporarily decreases the value of nodes being evaluated,
    encouraging parallel workers to explore different parts of the tree.
    """

    def __init__(self, game, model, args):
        self.game = game
        self.model = model
        self.args = args
        self.c_puct = args.get('c_puct', 1.0)
        self.num_simulations = args.get('num_simulations', 800)
        self.batch_size = args.get('batch_size', 16)
        self.virtual_loss = args.get('virtual_loss', 3)

    def select(self, node, virtual_losses):
        """Select child with highest UCB score, accounting for virtual losses."""
        total_visits = sum(child.visit_count for child in node.children.values())
        best_score = -float('inf')
        best_action = None

        for action, child in node.children.items():
            # Adjust value with virtual loss
            adjusted_value_sum = child.value_sum - virtual_losses.get(child, 0) * self.virtual_loss
            adjusted_visit_count = child.visit_count + virtual_losses.get(child, 0)
            q_value = -adjusted_value_sum / max(1, adjusted_visit_count)

            ucb_score = q_value + self.c_puct * child.prior * np.sqrt(total_visits) / (1 + adjusted_visit_count)
            if ucb_score > best_score:
                best_score = ucb_score
                best_action = action

        return best_action

    def run(self, state):
        """Run MCTS with virtual loss."""
        root = Node(0)
        root.state = state.clone()

        terminal_reward = self.game.reward(root.state)
        if terminal_reward is None:
            self._expand_node_single(root, root.state)
        else:
            return root

        num_batches = (self.num_simulations + self.batch_size - 1) // self.batch_size

        for batch_idx in range(num_batches):
            batch_size = min(self.batch_size, self.num_simulations - batch_idx * self.batch_size)
            self._run_batch_simulations(root, batch_size)

        return root

    def _run_batch_simulations(self, root, batch_size):
        """Run batch simulations with virtual loss."""
        virtual_losses = defaultdict(int)
        search_paths = []
        leaf_nodes = []
        leaf_states = []

        # Selection with virtual loss
        for _ in range(batch_size):
            node = root
            search_path = [node]

            while node.expanded():
                action = self.select(node, virtual_losses)
                node = node.children[action]
                search_path.append(node)
                virtual_losses[node] += 1

            search_paths.append(search_path)
            leaf_nodes.append(node)

            if node.state is None:
                parent_state = search_path[-2].state
                action = [a for a, child in search_path[-2].children.items() if child == node][0]
                leaf_state = self.game.get_next_state(parent_state, action)
                node.update_state(leaf_state)
            else:
                leaf_state = node.state

            leaf_states.append(leaf_state)

        # Evaluation
        terminal_values = []
        non_terminal_indices = []
        non_terminal_states = []
        non_terminal_nodes = []

        for i, state in enumerate(leaf_states):
            terminal_reward = self.game.reward(state)
            if terminal_reward is not None:
                terminal_values.append((i, terminal_reward))
            else:
                non_terminal_indices.append(i)
                non_terminal_states.append(state)
                non_terminal_nodes.append(leaf_nodes[i])

        # Batched prediction
        values = [None] * batch_size
        if non_terminal_states:
            states_tensor = torch.stack(non_terminal_states)
            policies, pred_values = self.model.predict_batch(states_tensor)

            for i, node_idx in enumerate(non_terminal_indices):
                node = non_terminal_nodes[i]
                state = non_terminal_states[i]
                policy = policies[i]
                value = pred_values[i].item()

                legal_actions = self.game.get_legal_actions(state)
                node.expand(legal_actions, policy.cpu())
                values[node_idx] = value

        for i, terminal_value in terminal_values:
            values[i] = terminal_value

        # Backpropagation (removing virtual losses)
        for search_path, value in zip(search_paths, values):
            if value is not None:
                for node in search_path:
                    virtual_losses[node] -= 1
                self._backpropagate(search_path, value)

    def _expand_node_single(self, node, state):
        """Expand a single node."""
        policy, value = self.model.predict_single(state)
        legal_actions = self.game.get_legal_actions(state)
        node.expand(legal_actions, policy.cpu())
        return value.item()

    @staticmethod
    def _backpropagate(search_path, value):
        """Backpropagate value."""
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += value
            value = -value
