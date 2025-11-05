# Bug Fix: Missing predict_single() Method in CachedBatchedModel

## Issue

The `CachedBatchedModel` class in `models/batched_model.py` was missing the `predict_single()` method, which is called by the `VirtualLossMCTS` and `ParallelMCTS` classes in their `_expand_node_single()` method.

This caused a runtime error when using `CachedBatchedModel` with `VirtualLossMCTS` as shown in the `training_optimized.ipynb` notebook:

```python
# This would fail with AttributeError
cached_model = CachedBatchedModel(model, device=device)
mcts = VirtualLossMCTS(game, cached_model, config)
root = mcts.run(state)  # Error: 'CachedBatchedModel' object has no attribute 'predict_single'
```

## Root Cause

The `CachedBatchedModel` class only implemented `predict_batch()` but not `predict_single()`, while the `BatchedModelWrapper` class had both methods. The MCTS implementations call `predict_single()` to expand the root node, causing the error.

## Fix

Added the `predict_single()` method to the `CachedBatchedModel` class with proper caching support:

```python
@torch.no_grad()
def predict_single(self, state):
    """
    Predict for a single state with caching.

    Args:
        state: (state_dim,) tensor

    Returns:
        policy: (num_actions,) tensor of action probabilities
        value: scalar tensor
    """
    key = self._state_to_key(state)

    # Check cache
    if key in self.cache:
        policy, value = self.cache[key]
        self.cache_hits += 1
        return policy.to(self.device), value.to(self.device)

    # Cache miss - predict
    self.cache_misses += 1
    state_batch = state.unsqueeze(0).to(self.device)
    with torch.no_grad():
        policy, value = self.model(state_batch)

    policy = policy.squeeze(0)
    value = value.squeeze(0)

    # Add to cache
    if len(self.cache) >= self.cache_size:
        self.cache.clear()

    self.cache[key] = (policy.cpu(), value.cpu())

    return policy, value
```

## Testing

Added a new test `test_cached_model_with_mcts()` in `test_optimizations.py` to verify that `CachedBatchedModel` works correctly with `VirtualLossMCTS`:

```python
def test_cached_model_with_mcts():
    """Test CachedBatchedModel with VirtualLossMCTS (notebook usage)."""
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
```

## Files Changed

1. **models/batched_model.py**
   - Added `predict_single()` method to `CachedBatchedModel` class

2. **test_optimizations.py**
   - Added `test_cached_model_with_mcts()` test function
   - Updated `run_all_tests()` to include the new test

## Impact

- Fixes the runtime error when using `CachedBatchedModel` with MCTS implementations
- The notebook `training_optimized.ipynb` now works correctly
- All existing functionality is preserved
- Cache statistics are properly tracked for single predictions

## Verification

The fix can be verified by running:

```bash
python test_optimizations.py
```

Or by running the notebook:

```bash
jupyter notebook training_optimized.ipynb
```

Both should now execute without errors.
