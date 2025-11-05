import torch
import torch.nn as nn
from collections import defaultdict


class BatchedModelWrapper:
    """
    Wrapper for batched model predictions to optimize MCTS.

    Instead of calling the model once per MCTS simulation, this wrapper collects
    multiple prediction requests and processes them in a single batch, significantly
    improving GPU utilization and overall throughput.
    """

    def __init__(self, model, device='cpu'):
        """
        Args:
            model: The neural network model (e.g., ChopsticksMLP)
            device: Device to run predictions on ('cpu' or 'cuda')
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def predict_batch(self, states):
        """
        Predict policy and value for a batch of states.

        Args:
            states: (batch_size, state_dim) tensor or list of tensors

        Returns:
            policies: (batch_size, num_actions) tensor of action probabilities
            values: (batch_size, 1) tensor of state values
        """
        if isinstance(states, list):
            states = torch.stack(states)

        states = states.to(self.device)

        with torch.no_grad():
            policies, values = self.model(states)

        return policies, values

    @torch.no_grad()
    def predict_single(self, state):
        """
        Predict for a single state (convenience method).

        Args:
            state: (state_dim,) tensor

        Returns:
            policy: (num_actions,) tensor of action probabilities
            value: scalar tensor
        """
        state = state.unsqueeze(0).to(self.device)
        with torch.no_grad():
            policy, value = self.model(state)
        return policy.squeeze(0), value.squeeze(0)


class CachedBatchedModel:
    """
    Batched model with caching for repeated state evaluations.

    During MCTS, the same state might be evaluated multiple times.
    This wrapper caches predictions to avoid redundant computations.
    """

    def __init__(self, model, device='cpu', cache_size=10000):
        """
        Args:
            model: The neural network model
            device: Device to run predictions on
            cache_size: Maximum number of cached predictions
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.cache = {}
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0

    def _state_to_key(self, state):
        """Convert state tensor to hashable key."""
        return tuple(state.cpu().numpy().round(4))

    @torch.no_grad()
    def predict_batch(self, states):
        """
        Predict with caching for a batch of states.

        Args:
            states: (batch_size, state_dim) tensor

        Returns:
            policies: (batch_size, num_actions) tensor
            values: (batch_size, 1) tensor
        """
        batch_size = states.shape[0]
        policies = []
        values = []
        uncached_indices = []
        uncached_states = []

        # Check cache for each state
        for i, state in enumerate(states):
            key = self._state_to_key(state)
            if key in self.cache:
                policy, value = self.cache[key]
                policies.append(policy)
                values.append(value)
                self.cache_hits += 1
            else:
                uncached_indices.append(i)
                uncached_states.append(state)
                self.cache_misses += 1

        # Predict uncached states in batch
        if uncached_states:
            uncached_states_tensor = torch.stack(uncached_states).to(self.device)
            with torch.no_grad():
                new_policies, new_values = self.model(uncached_states_tensor)

            # Add to cache and results
            for i, idx in enumerate(uncached_indices):
                policy = new_policies[i]
                value = new_values[i]
                key = self._state_to_key(states[idx])

                # Simple cache eviction: clear if too large
                if len(self.cache) >= self.cache_size:
                    self.cache.clear()

                self.cache[key] = (policy.cpu(), value.cpu())
                policies.insert(idx, policy)
                values.insert(idx, value)

        # Stack results
        policies = torch.stack([p.to(self.device) for p in policies])
        values = torch.stack([v.to(self.device) for v in values])

        return policies, values

    def clear_cache(self):
        """Clear the prediction cache."""
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0

    def get_cache_stats(self):
        """Get cache hit/miss statistics."""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0
        return {
            'hits': self.cache_hits,
            'misses': self.cache_misses,
            'hit_rate': hit_rate,
            'size': len(self.cache)
        }
