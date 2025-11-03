import torch
import numpy as np

DTYPE = torch.float32
REQUIRES_GRAD = False
NUM_ROUNDS_IDX = 4  # Index of number of rounds in state tensor
CURR_PLAYER_IDX = 5  # Index of current player in state tensor
NUM_ACTIONS = 10  # 6 tap actions + 4 split actions

class ChopsticksGame:
    '''
    Chopsticks game implementation for MCTS.
    Each play has two hands with 0-4 fingers with each hand starting at 1 finger.
    Players take turns to either tap opponent's hand or split their own fingers.
    When a hand reaches 5 or more fingers, it becomes 0 (dead).
    0 fingers means the hand is "dead" and cannot be used until split.
    You can tap your own hand to your other hand.
    You cannot tap a dead hand.
    You can only split if it results in a different configuration (mirror images not allowed).
    Splits must result in both hands having at least 1 finger.
    Winner is the player who makes the opponent lose both hands.
    Game is a draw if no winner is decided within a set number of moves.
    State is represented as a 1D torch tensor of int type: (p1_left, p1_right, p2_left, p2_right, current_player)
    current_player is 0 (p1) or 1 (p2) indicating whose turn it is.
    Actions are represented as integers from 0 to 9:
    0-7: Tap actions
        0: curr_left taps oppo_left
        1: curr_left taps oppo_right
        2: curr_left taps curr_right
        3: curr_right taps oppo_left
        4: curr_right taps oppo_right
        5: curr_right taps curr_left
    6-9: Split actions
        6: Split current fingers with left hand having 1 finger
        7: Split current fingers with left hand having 2 fingers
        8: Split current fingers with left hand having 3 fingers
        9: Split current fingers with left hand having 4 fingers
    '''
    def __init__(self, draw_limit=100):
        self.draw_limit = draw_limit
        self.state = torch.tensor([1, 1, 1, 1, 0, 0], dtype=DTYPE, requires_grad=REQUIRES_GRAD)

    def reset(self):
        self.state = torch.tensor([1, 1, 1, 1, 0, 0], dtype=DTYPE, requires_grad=REQUIRES_GRAD)
        return self.state

    def play(self, action):
        self.state = self.get_next_state(self.state, action)
        reward = self.reward(self.state)
        if reward is None:
            return True # Game continues
        else:
            return False # Game ended

    def state_dim(self):
        return self.state.shape[0]

    def num_actions(self):
        return NUM_ACTIONS  # 6 tap actions + 4 split actions

    def reward(self, state):
        '''
        Check if there is a winner in the current state.
        Returns:
            None if no winner,
            -1 if current player has lost i.e. opposing player has won,
        Since the game is ended after a player's winning move and
        the state switches to the other player,
        we return -1 to indicate the current player has lost.
        There are no draws in Chopsticks.
        '''
        curr = state[CURR_PLAYER_IDX].item()
        p1_hands = state[0:2]
        p2_hands = state[2:4]
        if p1_hands.sum() == 0:
            if curr == 0:
                return -1. # Player 2 wins
            else:
                raise ValueError("Invalid state: Player 1 has lost but it is not player 1's turn.")
        elif p2_hands.sum() == 0:
            if curr == 1:
                return -1.  # Player 1 wins
            else:
                raise ValueError("Invalid state: Player 2 has lost but it is not player 2's turn.")
        elif state[NUM_ROUNDS_IDX] == self.draw_limit:
            return 0.  # Draw
        else:
            return None  # No winner yet

    @staticmethod
    def get_legal_actions(state):
        fingers = state[:4].view((2,2))  # Reshape to 2x2 for easier indexing
        curr = int(state[CURR_PLAYER_IDX].item())
        legal_actions = []
        if not (fingers[curr,0] == 0 or fingers[1-curr,0] == 0):
            legal_actions.append(0)  # curr_left taps oppo_left
        if not (fingers[curr,0] == 0 or fingers[1-curr,1] == 0):
            legal_actions.append(1)  # curr_left taps oppo_right
        if not (fingers[curr,0] == 0 or fingers[curr,1] == 0):
            legal_actions.append(2)  # curr_left taps curr_right
        if not (fingers[curr,1] == 0 or fingers[1-curr,0] == 0):
            legal_actions.append(3)  # curr_right taps oppo_left
        if not (fingers[curr,1] == 0 or fingers[1-curr,1] == 0):
            legal_actions.append(4)  # curr_right taps oppo_right
        if not (fingers[curr,1] == 0 or fingers[curr,0] == 0):
            legal_actions.append(5)  # curr_right taps curr_left
        total = fingers[curr,0] + fingers[curr,1]
        for left in range(1,5):
            right = total - left
            if right <= 0 or right > 4:
                continue
            if left == fingers[curr,0] and right == fingers[curr,1]:
                continue  # No change
            if left == fingers[curr,1] and right == fingers[curr,0]:
                continue  # Mirror image

            legal_actions.append(5 + left)  # Split action

        return torch.tensor(legal_actions, dtype=torch.int32, requires_grad=REQUIRES_GRAD)

    @staticmethod
    def get_next_state(state, action):
        # Returns the next state after applying the action to the current state

        state = state.clone()  # Clone to avoid modifying original state
        fingers = state[:4].view((2,2))  # Reshape to 2x2 for easier indexing
        curr = int(state[CURR_PLAYER_IDX].item())
        if action == 0:
            fingers[1-curr,0] += fingers[curr,0]
        elif action == 1:
            fingers[1-curr,1] += fingers[curr,0]
        elif action == 2:
            fingers[curr,1] += fingers[curr,0]
        elif action == 3:
            fingers[1-curr,0] += fingers[curr,1]
        elif action == 4:
            fingers[1-curr,1] += fingers[curr,1]
        elif action == 5:
            fingers[curr,0] += fingers[curr,1]
        elif action in [6,7,8,9]:
            total = fingers[curr,0] + fingers[curr,1]
            left = action - 5
            right = total - left
            # no checking for legality as it is assumed action is legal
            fingers[curr,0] = left
            fingers[curr,1] = right
        state[:4] = fingers.view(-1)  # Update fingers and mod 5
        state[:4] = torch.minimum(state[:4], torch.tensor(5))  # Cap at 5 fingers
        state[:4] = state[:4] % 5
        state[NUM_ROUNDS_IDX] += 1  # Increment number of rounds
        state[CURR_PLAYER_IDX] = 1 - state[CURR_PLAYER_IDX]  # Switch current player
        return state

    @staticmethod
    def get_reward_for_player(state):
        # Returns the reward for the current player in the given state
        # Returns None if the game is not over
        pass

    @staticmethod
    def describe_action(action):
        """Return a human-readable description of the action."""
        action_descriptions = {
            0: "Current left hand taps opponent's left hand",
            1: "Current left hand taps opponent's right hand",
            2: "Current left hand taps current right hand",
            3: "Current right hand taps opponent's left hand",
            4: "Current right hand taps opponent's right hand",
            5: "Current right hand taps current left hand",
            6: "Split fingers: left hand gets 1 finger",
            7: "Split fingers: left hand gets 2 fingers",
            8: "Split fingers: left hand gets 3 fingers",
            9: "Split fingers: left hand gets 4 fingers"
        }
        return action_descriptions.get(action, "Unknown action")

    @staticmethod
    def print_state(state, label=""):
        """Print the state in a readable format."""
        p1_left, p1_right, p2_left, p2_right, num_rounds_played, curr_player = state.to(torch.int32).tolist()
        player = "P1" if curr_player == 0 else "P2"
        print(f"{label}")
        print(f"  Rounds Played: {num_rounds_played}")
        print(f"  P1: Left={p1_left}, Right={p1_right}")
        print(f"  P2: Left={p2_left}, Right={p2_right}")
        print(f"  Current Player: {player}")

    @staticmethod
    def print_winner_result(reward, state):
        """Print the winner result in a readable format."""
        if reward is None:
            print("  Winner: None (Game continues)")
        elif reward == -1:
            print(f"  Winner: Player {2 - state[CURR_PLAYER_IDX].item()}")


    def test_get_next_state(self):
        """Test the get_next_state function with various scenarios following game rules."""

        print("="*70)
        print("CHOPSTICKS GAME - get_next_state() TEST CASES")
        print("="*70)

        # Test 1: Initial game state - P1 left taps P2 left
        print("\n--- TEST 1: Initial state - P1 left taps P2 left (Action 0) ---")
        state = torch.tensor([1, 1, 1, 1, 0], dtype=torch.int32)
        self.print_state(state, "Initial State:")
        print(f"Action: {self.describe_action(0)}")
        state = ChopsticksGame.get_next_state(state, 0)
        self.print_state(state, "After Action:")

        # Test 2: Tap causing hand to reach exactly 5 (becomes 0/dead)
        print("\n--- TEST 2: Tap causing hand to die (3+2=5 -> 0) (Action 0) ---")
        state = torch.tensor([3, 2, 2, 1, 0], dtype=torch.int32)
        self.print_state(state, "Initial State:")
        print(f"Action: {self.describe_action(0)}")
        state = ChopsticksGame.get_next_state(state, 0)
        self.print_state(state, "After Action:")

        # Test 3: Tap causing hand to exceed 5 (e.g., 4+2=6 -> 1 after mod)
        print("\n--- TEST 3: Tap causing overflow (4+2=6 mod 5 = 1) (Action 4) ---")
        state = torch.tensor([2, 4, 1, 2, 0], dtype=torch.int32)
        self.print_state(state, "Initial State:")
        print(f"Action: {self.describe_action(4)}")
        state = ChopsticksGame.get_next_state(state, 4)
        self.print_state(state, "After Action:")

        # Test 4: Self-tap (current left taps current right)
        print("\n--- TEST 4: P1 left taps P1 right (Action 2) ---")
        state = torch.tensor([2, 1, 3, 2, 0], dtype=torch.int32)
        self.print_state(state, "Initial State:")
        print(f"Action: {self.describe_action(2)}")
        state = ChopsticksGame.get_next_state(state, 2)
        self.print_state(state, "After Action:")

        # Test 5: Self-tap resulting in death (3+2=5 -> 0)
        print("\n--- TEST 5: P2 right taps P2 left causing death (Action 5) ---")
        state = torch.tensor([2, 3, 3, 2, 1], dtype=torch.int32)
        self.print_state(state, "Initial State:")
        print(f"Action: {self.describe_action(5)}")
        state = ChopsticksGame.get_next_state(state, 5)
        self.print_state(state, "After Action:")

        # Test 6: Split action - even split (4 -> 2,2)
        print("\n--- TEST 6: P1 splits 4 fingers evenly (2-2) (Action 7) ---")
        state = torch.tensor([4, 0, 2, 3, 0], dtype=torch.int32)
        self.print_state(state, "Initial State:")
        print(f"Action: {self.describe_action(7)}")
        state = ChopsticksGame.get_next_state(state, 7)
        self.print_state(state, "After Action:")

        # Test 7: Split action - uneven split (4 -> 3,1)
        print("\n--- TEST 7: P2 splits 4 fingers (3-1) (Action 8) ---")
        state = torch.tensor([1, 2, 0, 4, 1], dtype=torch.int32)
        self.print_state(state, "Initial State:")
        print(f"Action: {self.describe_action(8)}")
        state = ChopsticksGame.get_next_state(state, 8)
        self.print_state(state, "After Action:")

        # Test 8: Split to resurrect dead hand (3 -> 1,2)
        print("\n--- TEST 8: P1 resurrects dead hand (3 -> 1,2) (Action 6) ---")
        state = torch.tensor([3, 0, 2, 1, 0], dtype=torch.int32)
        self.print_state(state, "Initial State:")
        print(f"Action: {self.describe_action(6)}")
        state = ChopsticksGame.get_next_state(state, 6)
        self.print_state(state, "After Action:")

        # Test 9: Tap causing large overflow (4+3=7 mod 5 = 2)
        print("\n--- TEST 9: Tap causing large overflow (4+3=7 mod 5 = 2) (Action 1) ---")
        state = torch.tensor([4, 2, 1, 3, 0], dtype=torch.int32)
        self.print_state(state, "Initial State:")
        print(f"Action: {self.describe_action(1)}")
        state = ChopsticksGame.get_next_state(state, 1)
        self.print_state(state, "After Action:")

        # Test 10: P2's turn - basic tap
        print("\n--- TEST 10: P2 left taps P1 left (Action 0) ---")
        state = torch.tensor([2, 3, 3, 2, 1], dtype=torch.int32)
        self.print_state(state, "Initial State:")
        print(f"Action: {self.describe_action(0)}")
        state = ChopsticksGame.get_next_state(state, 0)
        self.print_state(state, "After Action:")

        # Test 11: P2 taps P1, resulting in death
        print("\n--- TEST 11: P2 right taps P1 right causing death (Action 4) ---")
        state = torch.tensor([1, 2, 2, 3, 1], dtype=torch.int32)
        self.print_state(state, "Initial State:")
        print(f"Action: {self.describe_action(4)}")
        state = ChopsticksGame.get_next_state(state, 4)
        self.print_state(state, "After Action:")

        # Test 12: Split with maximum total (8 -> 4,4)
        print("\n--- TEST 12: P1 splits 8 fingers (4-4) (Action 9) ---")
        state = torch.tensor([4, 4, 1, 2, 0], dtype=torch.int32)
        self.print_state(state, "Initial State:")
        print(f"Action: {self.describe_action(9)}")
        state = ChopsticksGame.get_next_state(state, 9)
        self.print_state(state, "After Action:")

        # Test 13: P2 resurrects dead hand via split (4 -> 2,2)
        print("\n--- TEST 13: P2 resurrects dead hand (4 -> 2,2) (Action 7) ---")
        state = torch.tensor([2, 3, 0, 4, 1], dtype=torch.int32)
        self.print_state(state, "Initial State:")
        print(f"Action: {self.describe_action(7)}")
        state = ChopsticksGame.get_next_state(state, 7)
        self.print_state(state, "After Action:")

        # Test 14: Tap with one hand dead (only using live hand)
        print("\n--- TEST 14: P1 right taps P2 left with P1 left dead (Action 3) ---")
        state = torch.tensor([0, 3, 2, 1, 0], dtype=torch.int32)
        self.print_state(state, "Initial State:")
        print(f"Action: {self.describe_action(3)}")
        state = ChopsticksGame.get_next_state(state, 3)
        self.print_state(state, "After Action:")

        # Test 15: Edge case - tap resulting in exactly 4 (no overflow)
        print("\n--- TEST 15: Tap resulting in max (3+1=4) (Action 0) ---")
        state = torch.tensor([3, 2, 1, 2, 0], dtype=torch.int32)
        self.print_state(state, "Initial State:")
        print(f"Action: {self.describe_action(0)}")
        state = ChopsticksGame.get_next_state(state, 0)
        self.print_state(state, "After Action:")

        # Test 16: Split 6 fingers (6 -> 3,3)
        print("\n--- TEST 16: P2 splits 6 fingers (3-3) (Action 8) ---")
        state = torch.tensor([3, 2, 2, 4, 1], dtype=torch.int32)
        self.print_state(state, "Initial State:")
        print(f"Action: {self.describe_action(8)}")
        state = ChopsticksGame.get_next_state(state, 8)
        self.print_state(state, "After Action:")

        # Test 17: Split 5 fingers (5 -> 1,4)
        print("\n--- TEST 17: P1 splits 5 fingers (1-4) (Action 6) ---")
        state = torch.tensor([1, 4, 3, 2, 0], dtype=torch.int32)
        self.print_state(state, "Initial State:")
        print(f"Action: {self.describe_action(6)}")
        state = ChopsticksGame.get_next_state(state, 6)
        self.print_state(state, "After Action:")

        # Test 18: Tap causing opponent's hand to become 0 (winning move scenario)
        print("\n--- TEST 18: P1 left taps P2 right killing it (1+4=5 -> 0) (Action 1) ---")
        state = torch.tensor([1, 2, 0, 4, 0], dtype=torch.int32)
        self.print_state(state, "Initial State:")
        print(f"Action: {self.describe_action(1)}")
        state = ChopsticksGame.get_next_state(state, 1)
        self.print_state(state, "After Action:")

        # Test 19: Large overflow (4+4=8 mod 5 = 3)
        print("\n--- TEST 19: Large overflow (4+4=8 mod 5 = 3) (Action 2) ---")
        state = torch.tensor([4, 4, 2, 1, 0], dtype=torch.int32)
        self.print_state(state, "Initial State:")
        print(f"Action: {self.describe_action(2)}")
        state = ChopsticksGame.get_next_state(state, 2)
        self.print_state(state, "After Action:")

        # Test 20: Split 7 fingers (7 -> 4,3)
        print("\n--- TEST 20: P1 splits 7 fingers (4-3) (Action 9) ---")
        state = torch.tensor([3, 4, 1, 2, 0], dtype=torch.int32)
        self.print_state(state, "Initial State:")
        print(f"Action: {self.describe_action(9)}")
        state = ChopsticksGame.get_next_state(state, 9)
        self.print_state(state, "After Action:")

        print("\n" + "="*70)
        print("TEST SUITE COMPLETED")
        print("="*70)

    def test_get_legal_actions(self):
        """Test the get_legal_actions function with various scenarios."""

        print("="*70)
        print("CHOPSTICKS GAME - get_legal_actions() TEST CASES")
        print("="*70)

        # Test 1: Initial game state - all tap actions available
        print("\n--- TEST 1: Initial state - all hands alive ---")
        state = torch.tensor([1, 1, 1, 1, 0], dtype=torch.int32)
        self.print_state(state, "State:")
        legal_actions = ChopsticksGame.get_legal_actions(state)
        print(f"Legal actions: {legal_actions.tolist()}")
        print("Descriptions:")
        for action in legal_actions:
            print(f"  {action}: {self.describe_action(action.item())}")

        # Test 2: P1 left hand is dead
        print("\n--- TEST 2: P1 left hand is dead (0 fingers) ---")
        state = torch.tensor([0, 3, 2, 1, 0], dtype=torch.int32)
        self.print_state(state, "State:")
        legal_actions = ChopsticksGame.get_legal_actions(state)
        print(f"Legal actions: {legal_actions.tolist()}")
        print("Descriptions:")
        for action in legal_actions:
            print(f"  {action}: {self.describe_action(action.item())}")

        # Test 3: P1 right hand is dead
        print("\n--- TEST 3: P1 right hand is dead (0 fingers) ---")
        state = torch.tensor([3, 0, 2, 1, 0], dtype=torch.int32)
        self.print_state(state, "State:")
        legal_actions = ChopsticksGame.get_legal_actions(state)
        print(f"Legal actions: {legal_actions.tolist()}")
        print("Descriptions:")
        for action in legal_actions:
            print(f"  {action}: {self.describe_action(action.item())}")

        # Test 4: P2 left hand is dead
        print("\n--- TEST 4: P2 left hand is dead (0 fingers) ---")
        state = torch.tensor([2, 3, 0, 4, 1], dtype=torch.int32)
        self.print_state(state, "State:")
        legal_actions = ChopsticksGame.get_legal_actions(state)
        print(f"Legal actions: {legal_actions.tolist()}")
        print("Descriptions:")
        for action in legal_actions:
            print(f"  {action}: {self.describe_action(action.item())}")

        # Test 5: P2 right hand is dead
        print("\n--- TEST 5: P2 right hand is dead (0 fingers) ---")
        state = torch.tensor([2, 1, 3, 0, 1], dtype=torch.int32)
        self.print_state(state, "State:")
        legal_actions = ChopsticksGame.get_legal_actions(state)
        print(f"Legal actions: {legal_actions.tolist()}")
        print("Descriptions:")
        for action in legal_actions:
            print(f"  {action}: {self.describe_action(action.item())}")

        # Test 6: Opponent has one dead hand (P1's turn, P2 left is dead)
        print("\n--- TEST 6: P1's turn, P2 left hand is dead ---")
        state = torch.tensor([2, 3, 0, 2, 0], dtype=torch.int32)
        self.print_state(state, "State:")
        legal_actions = ChopsticksGame.get_legal_actions(state)
        print(f"Legal actions: {legal_actions.tolist()}")
        print("Descriptions:")
        for action in legal_actions:
            print(f"  {action}: {self.describe_action(action.item())}")

        # Test 7: Opponent has one dead hand (P1's turn, P2 right is dead)
        print("\n--- TEST 7: P1's turn, P2 right hand is dead ---")
        state = torch.tensor([2, 3, 2, 0, 0], dtype=torch.int32)
        self.print_state(state, "State:")
        legal_actions = ChopsticksGame.get_legal_actions(state)
        print(f"Legal actions: {legal_actions.tolist()}")
        print("Descriptions:")
        for action in legal_actions:
            print(f"  {action}: {self.describe_action(action.item())}")

        # Test 8: Both current player's hands alive, can split evenly
        print("\n--- TEST 8: P1 has 4 fingers total (can split 2-2) ---")
        state = torch.tensor([4, 0, 2, 1, 0], dtype=torch.int32)
        self.print_state(state, "State:")
        legal_actions = ChopsticksGame.get_legal_actions(state)
        print(f"Legal actions: {legal_actions.tolist()}")
        print("Descriptions:")
        for action in legal_actions:
            print(f"  {action}: {self.describe_action(action.item())}")

        # Test 9: Split with 5 total fingers
        print("\n--- TEST 9: P2 has 5 fingers total (can split various ways) ---")
        state = torch.tensor([2, 1, 1, 4, 1], dtype=torch.int32)
        self.print_state(state, "State:")
        legal_actions = ChopsticksGame.get_legal_actions(state)
        print(f"Legal actions: {legal_actions.tolist()}")
        print("Descriptions:")
        for action in legal_actions:
            print(f"  {action}: {self.describe_action(action.item())}")

        # Test 10: Split with 6 total fingers
        print("\n--- TEST 10: P1 has 6 fingers total (2,4) ---")
        state = torch.tensor([2, 4, 3, 1, 0], dtype=torch.int32)
        self.print_state(state, "State:")
        legal_actions = ChopsticksGame.get_legal_actions(state)
        print(f"Legal actions: {legal_actions.tolist()}")
        print("Descriptions:")
        for action in legal_actions:
            print(f"  {action}: {self.describe_action(action.item())}")

        # Test 11: Current player has symmetric hands (no new split possible)
        print("\n--- TEST 11: P1 has symmetric hands (2,2) - no valid splits ---")
        state = torch.tensor([2, 2, 3, 1, 0], dtype=torch.int32)
        self.print_state(state, "State:")
        legal_actions = ChopsticksGame.get_legal_actions(state)
        print(f"Legal actions: {legal_actions.tolist()}")
        print("Descriptions:")
        for action in legal_actions:
            print(f"  {action}: {self.describe_action(action.item())}")

        # Test 12: Split with 7 total fingers
        print("\n--- TEST 12: P2 has 7 fingers total (3,4) ---")
        state = torch.tensor([1, 2, 3, 4, 1], dtype=torch.int32)
        self.print_state(state, "State:")
        legal_actions = ChopsticksGame.get_legal_actions(state)
        print(f"Legal actions: {legal_actions.tolist()}")
        print("Descriptions:")
        for action in legal_actions:
            print(f"  {action}: {self.describe_action(action.item())}")

        # Test 13: Split with 8 total fingers (max)
        print("\n--- TEST 13: P1 has 8 fingers total (4,4) ---")
        state = torch.tensor([4, 4, 2, 1, 0], dtype=torch.int32)
        self.print_state(state, "State:")
        legal_actions = ChopsticksGame.get_legal_actions(state)
        print(f"Legal actions: {legal_actions.tolist()}")
        print("Descriptions:")
        for action in legal_actions:
            print(f"  {action}: {self.describe_action(action.item())}")

        # Test 14: Current player has 3 total fingers
        print("\n--- TEST 14: P1 has 3 fingers total (1,2) ---")
        state = torch.tensor([1, 2, 3, 4, 0], dtype=torch.int32)
        self.print_state(state, "State:")
        legal_actions = ChopsticksGame.get_legal_actions(state)
        print(f"Legal actions: {legal_actions.tolist()}")
        print("Descriptions:")
        for action in legal_actions:
            print(f"  {action}: {self.describe_action(action.item())}")

        # Test 15: Both opponent hands are alive, both current hands alive
        print("\n--- TEST 15: All hands alive with various finger counts ---")
        state = torch.tensor([3, 2, 4, 1, 0], dtype=torch.int32)
        self.print_state(state, "State:")
        legal_actions = ChopsticksGame.get_legal_actions(state)
        print(f"Legal actions: {legal_actions.tolist()}")
        print("Descriptions:")
        for action in legal_actions:
            print(f"  {action}: {self.describe_action(action.item())}")

        # Test 16: P2's turn with current configuration
        print("\n--- TEST 16: P2's turn with 1,3 fingers ---")
        state = torch.tensor([2, 3, 1, 3, 1], dtype=torch.int32)
        self.print_state(state, "State:")
        legal_actions = ChopsticksGame.get_legal_actions(state)
        print(f"Legal actions: {legal_actions.tolist()}")
        print("Descriptions:")
        for action in legal_actions:
            print(f"  {action}: {self.describe_action(action.item())}")

        # Test 17: Edge case - only one finger on each hand (cannot split)
        print("\n--- TEST 17: P1 has 1,1 (cannot split to different config) ---")
        state = torch.tensor([1, 1, 2, 3, 0], dtype=torch.int32)
        self.print_state(state, "State:")
        legal_actions = ChopsticksGame.get_legal_actions(state)
        print(f"Legal actions: {legal_actions.tolist()}")
        print("Descriptions:")
        for action in legal_actions:
            print(f"  {action}: {self.describe_action(action.item())}")

        # Test 18: Current player has 1,4 configuration
        print("\n--- TEST 18: P2 has 1,4 configuration ---")
        state = torch.tensor([3, 2, 1, 4, 1], dtype=torch.int32)
        self.print_state(state, "State:")
        legal_actions = ChopsticksGame.get_legal_actions(state)
        print(f"Legal actions: {legal_actions.tolist()}")
        print("Descriptions:")
        for action in legal_actions:
            print(f"  {action}: {self.describe_action(action.item())}")

        # Test 19: Both current player hands are dead (should not happen in valid game)
        print("\n--- TEST 19: Edge case - P1 has both hands dead (0,0) ---")
        state = torch.tensor([0, 0, 2, 3, 0], dtype=torch.int32)
        self.print_state(state, "State:")
        legal_actions = ChopsticksGame.get_legal_actions(state)
        print(f"Legal actions: {legal_actions.tolist()}")
        if len(legal_actions) == 0:
            print("No legal actions (game should be over)")
        else:
            print("Descriptions:")
            for action in legal_actions:
                print(f"  {action}: {self.describe_action(action.item())}")

        # Test 20: Current player can only split (both hands dead on opponent)
        print("\n--- TEST 20: P2's turn, P1 has both hands dead (0,0) ---")
        state = torch.tensor([0, 0, 2, 3, 1], dtype=torch.int32)
        self.print_state(state, "State:")
        legal_actions = ChopsticksGame.get_legal_actions(state)
        print(f"Legal actions: {legal_actions.tolist()}")
        print("Descriptions:")
        for action in legal_actions:
            print(f"  {action}: {self.describe_action(action.item())}")

        print("\n" + "="*70)
        print("TEST SUITE COMPLETED")
        print("="*70)

    def test_reward(self):
        """Test the reward function with various scenarios."""

        print("="*70)
        print("CHOPSTICKS GAME - reward() TEST CASES")
        print("="*70)

        # Test 1: Initial state - no winner
        print("\n--- TEST 1: Initial state - no winner ---")
        state = torch.tensor([1, 1, 1, 1, 0], dtype=torch.int32)
        self.print_state(state, "State:")
        reward = ChopsticksGame.reward(state)
        self.print_winner_result(reward, state)

        # Test 2: P1 kills P2's last hand - P1 wins
        print("\n--- TEST 2: P1 kills P2's last hand (winning move) ---")
        state = torch.tensor([1, 2, 0, 4, 0], dtype=torch.int32)
        self.print_state(state, "Initial State:")
        print(f"Action: {self.describe_action(1)}")
        state = ChopsticksGame.get_next_state(state, 1)
        self.print_state(state, "After Action:")
        reward = ChopsticksGame.reward(state)
        self.print_winner_result(reward, state)

        # Test 3: P2 kills P1's last hand - P2 wins
        print("\n--- TEST 3: P2 kills P1's last hand (winning move) ---")
        state = torch.tensor([0, 3, 2, 2, 1], dtype=torch.int32)
        self.print_state(state, "Initial State:")
        print(f"Action: {self.describe_action(4)}")
        state = ChopsticksGame.get_next_state(state, 4)
        self.print_state(state, "After Action:")
        reward = ChopsticksGame.reward(state)
        self.print_winner_result(reward, state)

        # Test 4: P1 kills one of P2's hands but P2 still has one alive - no winner
        print("\n--- TEST 4: P1 kills one hand but P2 still has another - no winner ---")
        state = torch.tensor([3, 1, 2, 1, 0], dtype=torch.int32)
        self.print_state(state, "Initial State:")
        print(f"Action: {self.describe_action(0)}")
        state = ChopsticksGame.get_next_state(state, 0)
        self.print_state(state, "After Action:")
        reward = ChopsticksGame.reward(state)
        self.print_winner_result(reward, state)

        # Test 5: P2 kills one of P1's hands but P1 still has one alive - no winner
        print("\n--- TEST 5: P2 kills one hand but P1 still has another - no winner ---")
        state = torch.tensor([2, 1, 3, 0, 1], dtype=torch.int32)
        self.print_state(state, "Initial State:")
        print(f"Action: {self.describe_action(0)}")
        state = ChopsticksGame.get_next_state(state, 0)
        self.print_state(state, "After Action:")
        reward = ChopsticksGame.reward(state)
        self.print_winner_result(reward, state)

        # Test 6: P1 kills both P2 hands in final move - P1 wins
        print("\n--- TEST 6: P1 kills P2's only remaining hand - P1 wins ---")
        state = torch.tensor([2, 3, 0, 3, 0], dtype=torch.int32)
        self.print_state(state, "Initial State:")
        print(f"Action: {self.describe_action(1)}")
        state = ChopsticksGame.get_next_state(state, 1)
        self.print_state(state, "After Action:")
        reward = ChopsticksGame.reward(state)
        self.print_winner_result(reward, state)

        # Test 7: P2 kills both P1 hands in final move - P2 wins
        print("\n--- TEST 7: P2 kills P1's only remaining hand - P2 wins ---")
        state = torch.tensor([0, 2, 3, 1, 1], dtype=torch.int32)
        self.print_state(state, "Initial State:")
        print(f"Action: {self.describe_action(4)}")
        state = ChopsticksGame.get_next_state(state, 4)
        self.print_state(state, "After Action:")
        reward = ChopsticksGame.reward(state)
        self.print_winner_result(reward, state)

        # Test 8: Game in progress with both players having both hands alive - no winner
        print("\n--- TEST 8: Both players have both hands alive - no winner ---")
        state = torch.tensor([2, 3, 3, 2, 0], dtype=torch.int32)
        self.print_state(state, "Initial State:")
        print(f"Action: {self.describe_action(0)}")
        state = ChopsticksGame.get_next_state(state, 0)
        self.print_state(state, "After Action:")
        reward = ChopsticksGame.reward(state)
        self.print_winner_result(reward, state)

        # Test 9: P1 has one hand dead but still alive - no winner
        print("\n--- TEST 9: P1 has one dead hand but still alive - no winner ---")
        state = torch.tensor([0, 3, 2, 1, 0], dtype=torch.int32)
        self.print_state(state, "Initial State:")
        print(f"Action: {self.describe_action(3)}")
        state = ChopsticksGame.get_next_state(state, 3)
        self.print_state(state, "After Action:")
        reward = ChopsticksGame.reward(state)
        self.print_winner_result(reward, state)

        # Test 10: P2 has one hand dead but still alive - no winner
        print("\n--- TEST 10: P2 has one dead hand but still alive - no winner ---")
        state = torch.tensor([2, 1, 0, 4, 1], dtype=torch.int32)
        self.print_state(state, "Initial State:")
        print(f"Action: {self.describe_action(3)}")
        state = ChopsticksGame.get_next_state(state, 3)
        self.print_state(state, "After Action:")
        reward = ChopsticksGame.reward(state)
        self.print_winner_result(reward, state)

        # Test 11: P1 causes overflow killing P2's hand and winning
        print("\n--- TEST 11: P1 causes overflow (4+4=8->3) then later wins ---")
        state = torch.tensor([4, 2, 0, 4, 0], dtype=torch.int32)
        self.print_state(state, "Initial State:")
        print(f"Action: {self.describe_action(1)}")
        state = ChopsticksGame.get_next_state(state, 1)
        self.print_state(state, "After Action:")
        reward = ChopsticksGame.reward(state)
        self.print_winner_result(reward, state)

        # Test 12: Self-tap scenario - no winner
        print("\n--- TEST 12: P1 self-taps - no winner ---")
        state = torch.tensor([2, 1, 3, 2, 0], dtype=torch.int32)
        self.print_state(state, "Initial State:")
        print(f"Action: {self.describe_action(2)}")
        state = ChopsticksGame.get_next_state(state, 2)
        self.print_state(state, "After Action:")
        reward = ChopsticksGame.reward(state)
        self.print_winner_result(reward, state)

        # Test 13: P2 self-tap resulting in death but not losing - no winner
        print("\n--- TEST 13: P2 self-taps killing own hand but not losing - no winner ---")
        state = torch.tensor([2, 3, 3, 2, 1], dtype=torch.int32)
        self.print_state(state, "Initial State:")
        print(f"Action: {self.describe_action(5)}")
        state = ChopsticksGame.get_next_state(state, 5)
        self.print_state(state, "After Action:")
        reward = ChopsticksGame.reward(state)
        self.print_winner_result(reward, state)

        # Test 14: P1 wins with exact hit (4+1=5->0)
        print("\n--- TEST 14: P1 wins with exact hit killing last hand ---")
        state = torch.tensor([4, 2, 0, 1, 0], dtype=torch.int32)
        self.print_state(state, "Initial State:")
        print(f"Action: {self.describe_action(1)}")
        state = ChopsticksGame.get_next_state(state, 1)
        self.print_state(state, "After Action:")
        reward = ChopsticksGame.reward(state)
        self.print_winner_result(reward, state)

        # Test 15: P2 wins with exact hit (3+2=5->0)
        print("\n--- TEST 15: P2 wins with exact hit killing last hand ---")
        state = torch.tensor([0, 2, 3, 1, 1], dtype=torch.int32)
        self.print_state(state, "Initial State:")
        print(f"Action: {self.describe_action(4)}")
        state = ChopsticksGame.get_next_state(state, 4)
        self.print_state(state, "After Action:")
        reward = ChopsticksGame.reward(state)
        self.print_winner_result(reward, state)

        # Test 16: Split action - no winner
        print("\n--- TEST 16: P1 splits fingers - no winner ---")
        state = torch.tensor([4, 0, 2, 3, 0], dtype=torch.int32)
        self.print_state(state, "Initial State:")
        print(f"Action: {self.describe_action(7)}")
        state = ChopsticksGame.get_next_state(state, 7)
        self.print_state(state, "After Action:")
        reward = ChopsticksGame.reward(state)
        self.print_winner_result(reward, state)

        # Test 17: P2 split resurrects hand - no winner
        print("\n--- TEST 17: P2 resurrects dead hand via split - no winner ---")
        state = torch.tensor([2, 3, 0, 4, 1], dtype=torch.int32)
        self.print_state(state, "Initial State:")
        print(f"Action: {self.describe_action(7)}")
        state = ChopsticksGame.get_next_state(state, 7)
        self.print_state(state, "After Action:")
        reward = ChopsticksGame.reward(state)
        self.print_winner_result(reward, state)

        # Test 18: Regular tap that doesn't kill - no winner
        print("\n--- TEST 18: Regular tap with no kill - no winner ---")
        state = torch.tensor([1, 1, 1, 1, 0], dtype=torch.int32)
        self.print_state(state, "Initial State:")
        print(f"Action: {self.describe_action(0)}")
        state = ChopsticksGame.get_next_state(state, 0)
        self.print_state(state, "After Action:")
        reward = ChopsticksGame.reward(state)
        self.print_winner_result(reward, state)

        # Test 19: P1 kills P2's second-to-last hand - no winner yet
        print("\n--- TEST 19: P1 kills one hand when opponent has two - no winner ---")
        state = torch.tensor([3, 2, 2, 1, 0], dtype=torch.int32)
        self.print_state(state, "Initial State:")
        print(f"Action: {self.describe_action(0)}")
        state = ChopsticksGame.get_next_state(state, 0)
        self.print_state(state, "After Action:")
        reward = ChopsticksGame.reward(state)
        self.print_winner_result(reward, state)

        # Test 20: P1 makes winning move from strong position
        print("\n--- TEST 20: P1 wins from strong position ---")
        state = torch.tensor([3, 4, 0, 2, 0], dtype=torch.int32)
        self.print_state(state, "Initial State:")
        print(f"Action: {self.describe_action(1)}")
        state = ChopsticksGame.get_next_state(state, 1)
        self.print_state(state, "After Action:")
        reward = ChopsticksGame.reward(state)
        self.print_winner_result(reward, state)

        print("\n" + "="*70)
        print("TEST SUITE COMPLETED")
        print("="*70)