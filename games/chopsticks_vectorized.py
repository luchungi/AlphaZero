import torch
import numpy as np

DTYPE = torch.float32
REQUIRES_GRAD = False
NUM_ROUNDS_IDX = 4  # Index of number of rounds in state tensor
CURR_PLAYER_IDX = 5  # Index of current player in state tensor
NUM_ACTIONS = 10  # 6 tap actions + 4 split actions


class ChopsticksVectorized:
    '''
    Vectorized Chopsticks game implementation for parallel game execution.
    Supports running multiple game instances simultaneously using batched operations.

    State shape: (batch_size, 6) where each state is:
        [p1_left, p1_right, p2_left, p2_right, num_rounds, curr_player]

    This allows efficient parallel execution of multiple games.
    '''

    def __init__(self, batch_size, draw_limit=100, device='cpu'):
        self.batch_size = batch_size
        self.draw_limit = draw_limit
        self.device = device
        # Initialize batch of games
        self.states = torch.ones((batch_size, 6), dtype=DTYPE, device=device)
        self.states[:, NUM_ROUNDS_IDX] = 0  # Set rounds to 0
        self.states[:, CURR_PLAYER_IDX] = 0  # Set current player to 0

    def reset(self):
        """Reset all games in the batch."""
        self.states = torch.ones((self.batch_size, 6), dtype=DTYPE, device=self.device)
        self.states[:, NUM_ROUNDS_IDX] = 0
        self.states[:, CURR_PLAYER_IDX] = 0
        return self.states

    def reset_games(self, game_indices):
        """Reset specific games by their indices."""
        if len(game_indices) > 0:
            self.states[game_indices] = 1.0
            self.states[game_indices, NUM_ROUNDS_IDX] = 0
            self.states[game_indices, CURR_PLAYER_IDX] = 0

    @staticmethod
    def get_next_state_batch(states, actions):
        """
        Apply actions to a batch of states.

        Args:
            states: (batch_size, 6) tensor of game states
            actions: (batch_size,) tensor of actions to apply

        Returns:
            (batch_size, 6) tensor of next states
        """
        batch_size = states.shape[0]
        next_states = states.clone()
        fingers = next_states[:, :4].view(batch_size, 2, 2)  # (batch, player, hand)
        curr_players = next_states[:, CURR_PLAYER_IDX].long()

        # Create masks for each action type
        for action_idx in range(NUM_ACTIONS):
            mask = (actions == action_idx)
            if not mask.any():
                continue

            if action_idx == 0:  # curr_left taps oppo_left
                curr_idx = curr_players[mask]
                oppo_idx = 1 - curr_idx
                fingers[mask, oppo_idx, 0] += fingers[mask, curr_idx, 0]
            elif action_idx == 1:  # curr_left taps oppo_right
                curr_idx = curr_players[mask]
                oppo_idx = 1 - curr_idx
                fingers[mask, oppo_idx, 1] += fingers[mask, curr_idx, 0]
            elif action_idx == 2:  # curr_left taps curr_right
                curr_idx = curr_players[mask]
                fingers[mask, curr_idx, 1] += fingers[mask, curr_idx, 0]
            elif action_idx == 3:  # curr_right taps oppo_left
                curr_idx = curr_players[mask]
                oppo_idx = 1 - curr_idx
                fingers[mask, oppo_idx, 0] += fingers[mask, curr_idx, 1]
            elif action_idx == 4:  # curr_right taps oppo_right
                curr_idx = curr_players[mask]
                oppo_idx = 1 - curr_idx
                fingers[mask, oppo_idx, 1] += fingers[mask, curr_idx, 1]
            elif action_idx == 5:  # curr_right taps curr_left
                curr_idx = curr_players[mask]
                fingers[mask, curr_idx, 0] += fingers[mask, curr_idx, 1]
            elif action_idx >= 6:  # Split actions
                curr_idx = curr_players[mask]
                total = fingers[mask, curr_idx, 0] + fingers[mask, curr_idx, 1]
                left = action_idx - 5
                right = total - left
                fingers[mask, curr_idx, 0] = left
                fingers[mask, curr_idx, 1] = right

        # Update state with finger counts and apply mod 5
        next_states[:, :4] = fingers.view(batch_size, 4)
        next_states[:, :4] = torch.clamp(next_states[:, :4], max=5)
        next_states[:, :4] = next_states[:, :4] % 5

        # Increment rounds and switch players
        next_states[:, NUM_ROUNDS_IDX] += 1
        next_states[:, CURR_PLAYER_IDX] = 1 - next_states[:, CURR_PLAYER_IDX]

        return next_states

    @staticmethod
    def get_legal_actions_batch(states):
        """
        Get legal actions for a batch of states.

        Returns a list of tensors, one per state, containing legal action indices.
        """
        batch_size = states.shape[0]
        legal_actions_list = []

        for i in range(batch_size):
            state = states[i]
            fingers = state[:4].view(2, 2)
            curr = int(state[CURR_PLAYER_IDX].item())
            legal_actions = []

            # Tap actions
            if not (fingers[curr, 0] == 0 or fingers[1-curr, 0] == 0):
                legal_actions.append(0)
            if not (fingers[curr, 0] == 0 or fingers[1-curr, 1] == 0):
                legal_actions.append(1)
            if not (fingers[curr, 0] == 0 or fingers[curr, 1] == 0):
                legal_actions.append(2)
            if not (fingers[curr, 1] == 0 or fingers[1-curr, 0] == 0):
                legal_actions.append(3)
            if not (fingers[curr, 1] == 0 or fingers[1-curr, 1] == 0):
                legal_actions.append(4)
            if not (fingers[curr, 1] == 0 or fingers[curr, 0] == 0):
                legal_actions.append(5)

            # Split actions
            total = fingers[curr, 0] + fingers[curr, 1]
            for left in range(1, 5):
                right = total - left
                if right <= 0 or right > 4:
                    continue
                if left == fingers[curr, 0] and right == fingers[curr, 1]:
                    continue  # No change
                if left == fingers[curr, 1] and right == fingers[curr, 0]:
                    continue  # Mirror image
                legal_actions.append(5 + left)

            legal_actions_list.append(torch.tensor(legal_actions, dtype=torch.long, device=states.device))

        return legal_actions_list

    @staticmethod
    def get_rewards_batch(states, draw_limit=100):
        """
        Get rewards for a batch of states.

        Returns:
            rewards: (batch_size,) tensor where:
                None placeholder (use torch.nan) if game continues
                -1.0 if current player has lost
                0.0 if draw
        """
        batch_size = states.shape[0]
        rewards = torch.full((batch_size,), float('nan'), dtype=DTYPE, device=states.device)

        curr_players = states[:, CURR_PLAYER_IDX]
        p1_hands = states[:, 0:2].sum(dim=1)
        p2_hands = states[:, 2:4].sum(dim=1)
        rounds = states[:, NUM_ROUNDS_IDX]

        # Player 1 lost (current player is P1)
        p1_lost_mask = (p1_hands == 0) & (curr_players == 0)
        rewards[p1_lost_mask] = -1.0

        # Player 2 lost (current player is P2)
        p2_lost_mask = (p2_hands == 0) & (curr_players == 1)
        rewards[p2_lost_mask] = -1.0

        # Draw
        draw_mask = (rounds >= draw_limit)
        rewards[draw_mask] = 0.0

        return rewards

    @staticmethod
    def is_terminal_batch(states, draw_limit=100):
        """Check which games in the batch are terminal."""
        rewards = ChopsticksVectorized.get_rewards_batch(states, draw_limit)
        return ~torch.isnan(rewards)

    def play_batch(self, actions):
        """
        Play actions for all games in the batch.

        Args:
            actions: (batch_size,) tensor of actions

        Returns:
            (batch_size,) boolean tensor indicating which games continue
        """
        self.states = self.get_next_state_batch(self.states, actions)
        rewards = self.get_rewards_batch(self.states, self.draw_limit)
        return torch.isnan(rewards)  # True if game continues


# Single-state utility functions for compatibility with original interface
def get_legal_actions(state):
    """Get legal actions for a single state (compatible with original interface)."""
    return ChopsticksVectorized.get_legal_actions_batch(state.unsqueeze(0))[0]


def get_next_state(state, action):
    """Get next state for a single state-action pair (compatible with original interface)."""
    if isinstance(action, int):
        action = torch.tensor([action], dtype=torch.long, device=state.device)
    else:
        action = action.unsqueeze(0) if action.dim() == 0 else action
    return ChopsticksVectorized.get_next_state_batch(state.unsqueeze(0), action)[0]


def reward(state, draw_limit=100):
    """Get reward for a single state (compatible with original interface)."""
    r = ChopsticksVectorized.get_rewards_batch(state.unsqueeze(0), draw_limit)[0]
    return None if torch.isnan(r) else r.item()
