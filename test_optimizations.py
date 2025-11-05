"""
Test script to verify optimized implementations work correctly.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time

# Import game implementations
from games.chopsticks import ChopsticksGame
from games.chopsticks_vectorized import ChopsticksVectorized

# Import models
from models.chopsticks import ChopsticksMLP
from models.batched_model import BatchedModelWrapper, CachedBatchedModel

# Import MCTS implementations
from sims.tree import MCTS
from sims.tree_parallel import ParallelMCTS, VirtualLossMCTS

# Import trainers
from utils.trainer import self_play, create_dataloader, train_model
from utils.trainer_parallel import self_play_parallel


def test_vectorized_game():
    """Test vectorized game implementation."""
    print("Testing vectorized game...")

    batch_size = 4
    vec_game = ChopsticksVectorized(batch_size=batch_size, device='cpu')

    # Test reset
    states = vec_game.reset()
    assert states.shape == (batch_size, 6), f"Expected shape (4, 6), got {states.shape}"

    # Test get_next_state_batch
    actions = torch.tensor([0, 1, 2, 3])
    next_states = vec_game.get_next_state_batch(states, actions)
    assert next_states.shape == (batch_size, 6)

    # Test get_legal_actions_batch
    legal_actions = vec_game.get_legal_actions_batch(states)
    assert len(legal_actions) == batch_size

    # Test get_rewards_batch
    rewards = vec_game.get_rewards_batch(states)
    assert rewards.shape == (batch_size,)

    print("✓ Vectorized game tests passed")


def test_batched_model():
    """Test batched model wrapper."""
    print("\nTesting batched model...")

    # Create model
    model = ChopsticksMLP(input_size=6, output_size=10, hidden_size=64)
    batched_model = BatchedModelWrapper(model, device='cpu')

    # Test batch prediction
    batch_size = 8
    states = torch.ones((batch_size, 6))
    policies, values = batched_model.predict_batch(states)

    assert policies.shape == (batch_size, 10), f"Expected policies shape (8, 10), got {policies.shape}"
    assert values.shape == (batch_size, 1), f"Expected values shape (8, 1), got {values.shape}"

    # Test single prediction
    state = torch.ones(6)
    policy, value = batched_model.predict_single(state)
    assert policy.shape == (10,)
    assert value.shape == (1,)

    print("✓ Batched model tests passed")


def test_cached_model():
    """Test cached batched model."""
    print("\nTesting cached model...")

    model = ChopsticksMLP(input_size=6, output_size=10, hidden_size=64)
    cached_model = CachedBatchedModel(model, device='cpu', cache_size=100)

    # First prediction (cache miss)
    state = torch.ones(6)
    policy1, value1 = cached_model.predict_single(state)

    # Second prediction (cache hit)
    policy2, value2 = cached_model.predict_single(state)

    # Should be identical
    assert torch.allclose(policy1, policy2)
    assert torch.allclose(value1, value2)

    # Check cache stats
    stats = cached_model.get_cache_stats()
    assert stats['hits'] >= 1, "Expected at least 1 cache hit"

    print(f"✓ Cached model tests passed (hit rate: {stats['hit_rate']:.2%})")


def test_parallel_mcts():
    """Test parallel MCTS implementation."""
    print("\nTesting parallel MCTS...")

    # Setup
    game = ChopsticksGame()
    model = ChopsticksMLP(input_size=6, output_size=10, hidden_size=64)
    batched_model = BatchedModelWrapper(model, device='cpu')

    config = {
        'c_puct': 1.0,
        'num_simulations': 50,  # Reduced for testing
        'batch_size': 8,
    }

    mcts = ParallelMCTS(game, batched_model, config)

    # Run MCTS
    state = game.reset()
    start = time.time()
    root = mcts.run(state)
    elapsed = time.time() - start

    assert root.visit_count > 0, "Root should have visits"
    assert len(root.children) > 0, "Root should have children"

    print(f"✓ Parallel MCTS tests passed ({elapsed:.2f}s for 50 simulations)")


def test_virtual_loss_mcts():
    """Test MCTS with virtual loss."""
    print("\nTesting virtual loss MCTS...")

    game = ChopsticksGame()
    model = ChopsticksMLP(input_size=6, output_size=10, hidden_size=64)
    batched_model = BatchedModelWrapper(model, device='cpu')

    config = {
        'c_puct': 1.0,
        'num_simulations': 50,
        'batch_size': 8,
        'virtual_loss': 3,
    }

    mcts = VirtualLossMCTS(game, batched_model, config)

    state = game.reset()
    root = mcts.run(state)

    assert root.visit_count > 0
    assert len(root.children) > 0

    print("✓ Virtual loss MCTS tests passed")


def test_cached_model_with_mcts():
    """Test CachedBatchedModel with VirtualLossMCTS (notebook usage)."""
    print("\nTesting CachedBatchedModel with VirtualLossMCTS...")

    game = ChopsticksGame()
    model = ChopsticksMLP(input_size=6, output_size=10, hidden_size=64)
    cached_model = CachedBatchedModel(model, device='cpu', cache_size=1000)

    config = {
        'c_puct': 1.0,
        'num_simulations': 50,
        'batch_size': 8,
        'virtual_loss': 3,
    }

    mcts = VirtualLossMCTS(game, cached_model, config)

    state = game.reset()
    root = mcts.run(state)

    assert root.visit_count > 0
    assert len(root.children) > 0

    # Check cache is being used
    stats = cached_model.get_cache_stats()
    assert stats['size'] > 0, "Cache should have entries"

    print(f"✓ CachedBatchedModel with MCTS tests passed (cache size: {stats['size']})")


def test_self_play_parallel():
    """Test parallel self-play."""
    print("\nTesting parallel self-play...")

    game = ChopsticksGame()
    model = ChopsticksMLP(input_size=6, output_size=10, hidden_size=64)
    batched_model = BatchedModelWrapper(model, device='cpu')

    config = {
        'c_puct': 1.0,
        'num_simulations': 20,  # Very low for testing
        'batch_size': 4,
    }

    mcts = ParallelMCTS(game, batched_model, config)

    # Run self-play
    start = time.time()
    samples = self_play_parallel(
        mcts, game,
        num_games=3,
        num_workers=2,
        show_progress=False
    )
    elapsed = time.time() - start

    assert len(samples) > 0, "Should generate samples"

    # Check sample format
    state, action_probs, reward = samples[0]
    assert state.shape == (6,)
    assert action_probs.shape == (10,)
    assert isinstance(reward.item(), float)

    print(f"✓ Parallel self-play tests passed ({len(samples)} samples in {elapsed:.2f}s)")


def test_training_dataloader():
    """Test dataloader creation."""
    print("\nTesting dataloader creation...")

    # Create dummy samples
    samples = []
    for _ in range(20):
        state = torch.rand(6)
        action_probs = torch.rand(10)
        action_probs = action_probs / action_probs.sum()
        reward = torch.tensor(1.0)
        samples.append((state, action_probs, reward))

    # Create dataloader
    from utils.trainer_parallel import create_dataloader
    dataloader = create_dataloader(samples, batch_size=8, device='cpu')

    # Test iteration
    for states, action_probs, rewards in dataloader:
        assert states.shape[0] <= 8  # Batch size
        assert states.shape[1] == 6  # State dim
        assert action_probs.shape[1] == 10  # Action dim
        break

    print("✓ Dataloader tests passed")


def benchmark_mcts():
    """Benchmark sequential vs parallel MCTS."""
    print("\n" + "="*60)
    print("MCTS Performance Benchmark")
    print("="*60)

    game = ChopsticksGame()
    model = ChopsticksMLP(input_size=6, output_size=10, hidden_size=64)

    num_runs = 3
    num_sims = 100

    # Sequential MCTS
    print(f"\nSequential MCTS ({num_sims} simulations)...")
    config_seq = {
        'c_puct': 1.0,
        'num_simulations': num_sims,
    }
    mcts_seq = MCTS(game, model, config_seq)

    start = time.time()
    for _ in range(num_runs):
        root = mcts_seq.run(game.reset())
    seq_time = time.time() - start
    seq_avg = seq_time / num_runs
    print(f"  Average: {seq_avg:.3f}s per run")

    # Parallel MCTS
    print(f"\nParallel MCTS ({num_sims} simulations, batch=8)...")
    batched_model = BatchedModelWrapper(model, device='cpu')
    config_par = {
        'c_puct': 1.0,
        'num_simulations': num_sims,
        'batch_size': 8,
    }
    mcts_par = ParallelMCTS(game, batched_model, config_par)

    start = time.time()
    for _ in range(num_runs):
        root = mcts_par.run(game.reset())
    par_time = time.time() - start
    par_avg = par_time / num_runs
    print(f"  Average: {par_avg:.3f}s per run")

    # Results
    speedup = seq_time / par_time
    print(f"\nSpeedup: {speedup:.2f}x")
    print(f"Time saved: {seq_time - par_time:.2f}s ({(1 - par_time/seq_time)*100:.1f}%)")


def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("Running AlphaZero Optimization Tests")
    print("="*60)

    try:
        test_vectorized_game()
        test_batched_model()
        test_cached_model()
        test_parallel_mcts()
        test_virtual_loss_mcts()
        test_cached_model_with_mcts()
        test_self_play_parallel()
        test_training_dataloader()

        print("\n" + "="*60)
        print("All tests passed! ✓")
        print("="*60)

        # Run benchmark
        benchmark_mcts()

        return True

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
