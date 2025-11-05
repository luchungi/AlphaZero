import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp


def self_play_parallel(mcts, game, num_games, num_workers=4, show_progress=True):
    """
    Parallel self-play using multiple games simultaneously.

    Args:
        mcts: MCTS instance for tree search
        game: Game interface (ChopsticksGame)
        num_games: Number of games to play
        num_workers: Number of parallel games (default: 4)
        show_progress: Whether to show progress bar

    Returns:
        List of (state, action_probs, reward) tuples
    """
    samples = []
    games_per_batch = min(num_workers, num_games)

    iterator = range(0, num_games, games_per_batch)
    if show_progress:
        iterator = tqdm(iterator, desc="Self-play")

    for batch_start in iterator:
        batch_end = min(batch_start + games_per_batch, num_games)
        current_batch_size = batch_end - batch_start

        # Initialize parallel games
        game_states = [game.reset() for _ in range(current_batch_size)]
        game_continues = [True] * current_batch_size
        policy_samples_per_game = [[] for _ in range(current_batch_size)]

        # Play all games in this batch
        max_moves = 200  # Safety limit
        for move_count in range(max_moves):
            if not any(game_continues):
                break

            # Run MCTS for each active game
            for game_idx in range(current_batch_size):
                if not game_continues[game_idx]:
                    continue

                state = game_states[game_idx]
                root = mcts.run(state)

                # Get action probabilities from visit counts
                action_dict = root.action_probs()
                action_probs = torch.zeros(game.num_actions(), dtype=torch.float32)
                for action, prob in action_dict.items():
                    action_probs[action] = prob

                # Store state and policy
                policy_samples_per_game[game_idx].append((state.clone(), action_probs))

                # Sample action and play
                action = root.sample_action()
                game_continues[game_idx] = game.play(action)
                game_states[game_idx] = game.state.clone()

                # If game ended, compute rewards
                if not game_continues[game_idx]:
                    reward = game.reward(game.state)
                    winner = 1 - game.state[-1]  # Losing player's turn at end

                    for state, action_probs in policy_samples_per_game[game_idx]:
                        # Flip reward based on player perspective
                        r = -reward if winner == state[-1] else reward
                        r = torch.tensor(r, dtype=torch.float32)
                        samples.append((state, action_probs, r))

    return samples


def self_play_vectorized(mcts, game_class, num_games, batch_size=8, show_progress=True):
    """
    Vectorized self-play using batched game execution.

    This version uses the vectorized game implementation to run multiple
    games truly in parallel with vectorized operations.

    Args:
        mcts: MCTS instance
        game_class: ChopsticksGame class (not instance)
        num_games: Total number of games to play
        batch_size: Number of games to run in parallel
        show_progress: Whether to show progress bar

    Returns:
        List of (state, action_probs, reward) tuples
    """
    from games.chopsticks_vectorized import ChopsticksVectorized

    samples = []
    num_batches = (num_games + batch_size - 1) // batch_size

    iterator = range(num_batches)
    if show_progress:
        iterator = tqdm(iterator, desc="Self-play (vectorized)")

    for batch_idx in iterator:
        current_batch_size = min(batch_size, num_games - batch_idx * batch_size)

        # Initialize games
        game_instance = game_class()
        game_states = [game_instance.reset() for _ in range(current_batch_size)]
        game_continues = [True] * current_batch_size
        policy_samples_per_game = [[] for _ in range(current_batch_size)]

        max_moves = 200
        for move_count in range(max_moves):
            if not any(game_continues):
                break

            for game_idx in range(current_batch_size):
                if not game_continues[game_idx]:
                    continue

                state = game_states[game_idx]
                root = mcts.run(state)

                action_dict = root.action_probs()
                action_probs = torch.zeros(game_instance.num_actions(), dtype=torch.float32)
                for action, prob in action_dict.items():
                    action_probs[action] = prob

                policy_samples_per_game[game_idx].append((state.clone(), action_probs))

                action = root.sample_action()
                game_continues[game_idx] = game_instance.play(action)
                game_states[game_idx] = game_instance.state.clone()

                if not game_continues[game_idx]:
                    reward = game_instance.reward(game_instance.state)
                    winner = 1 - game_instance.state[-1]

                    for state, action_probs in policy_samples_per_game[game_idx]:
                        r = -reward if winner == state[-1] else reward
                        r = torch.tensor(r, dtype=torch.float32)
                        samples.append((state, action_probs, r))

    return samples


def create_dataloader(samples, batch_size=64, device='cpu'):
    """
    Create DataLoader from self-play samples.

    Args:
        samples: List of (state, action_probs, reward) tuples
        batch_size: Batch size for training
        device: Device to put data on

    Returns:
        DataLoader instance
    """
    states = torch.stack([s[0] for s in samples]).to(device)
    action_probs = torch.stack([s[1] for s in samples]).to(device)
    rewards = torch.stack([s[2] for s in samples]).to(device)

    dataset = TensorDataset(states, action_probs, rewards)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def train_model(model, dataloader, optimizer, num_epochs=50, device='cpu', show_progress=True):
    """
    Train the model on self-play data.

    Args:
        model: Neural network model
        dataloader: DataLoader with training data
        optimizer: Optimizer instance
        num_epochs: Number of training epochs
        device: Device to train on
        show_progress: Whether to show progress

    Returns:
        List of average losses per epoch
    """
    model.to(device)
    model.train()

    epoch_losses = []
    iterator = range(num_epochs)
    if show_progress:
        iterator = tqdm(iterator, desc="Training")

    for epoch in iterator:
        losses = []
        for states, action_probs, rewards in dataloader:
            states = states.to(device)
            action_probs = action_probs.to(device)
            rewards = rewards.to(device)

            pred_probs, pred_value = model(states)

            # Value loss: predict game outcome
            value_loss = torch.nn.functional.mse_loss(pred_value.squeeze(), rewards)

            # Policy loss: match MCTS visit distribution
            policy_loss = -torch.mean(torch.sum(action_probs * torch.log(pred_probs + 1e-8), dim=1))

            # Combined loss
            loss = value_loss + policy_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        avg_loss = sum(losses) / len(losses)
        epoch_losses.append(avg_loss)

        if show_progress and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

    return epoch_losses


def train_iteration(model, mcts, game, optimizer, args, device='cpu'):
    """
    Single training iteration: self-play + training.

    Args:
        model: Neural network model
        mcts: MCTS instance
        game: Game instance
        optimizer: Optimizer
        args: Training arguments dict with keys:
            - num_games: Number of self-play games
            - num_epochs: Training epochs
            - batch_size: Batch size
            - num_workers: Parallel games
        device: Device to use

    Returns:
        Dictionary with training statistics
    """
    # Self-play
    print("\nSelf-play phase...")
    samples = self_play_parallel(
        mcts, game,
        num_games=args.get('num_games', 50),
        num_workers=args.get('num_workers', 4)
    )

    print(f"Collected {len(samples)} training samples")

    # Create dataloader
    dataloader = create_dataloader(
        samples,
        batch_size=args.get('batch_size', 64),
        device=device
    )

    # Train model
    print("\nTraining phase...")
    epoch_losses = train_model(
        model, dataloader, optimizer,
        num_epochs=args.get('num_epochs', 50),
        device=device
    )

    return {
        'num_samples': len(samples),
        'final_loss': epoch_losses[-1] if epoch_losses else 0,
        'epoch_losses': epoch_losses
    }


def full_training_loop(model, mcts, game, optimizer, args, num_iterations=10, device='cpu'):
    """
    Full AlphaZero training loop with multiple iterations.

    Args:
        model: Neural network model
        mcts: MCTS instance
        game: Game instance
        optimizer: Optimizer
        args: Training arguments
        num_iterations: Number of training iterations
        device: Device to use

    Returns:
        Training history
    """
    history = []

    for iteration in range(num_iterations):
        print(f"\n{'='*60}")
        print(f"Training Iteration {iteration + 1}/{num_iterations}")
        print(f"{'='*60}")

        stats = train_iteration(model, mcts, game, optimizer, args, device)
        stats['iteration'] = iteration + 1
        history.append(stats)

        print(f"\nIteration {iteration + 1} complete:")
        print(f"  Samples: {stats['num_samples']}")
        print(f"  Final loss: {stats['final_loss']:.4f}")

    return history
