
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

def self_play(mcts, game, num_games):
    samples = []
    for _ in tqdm(range(num_games)):
        game.reset()
        game_continues = True
        policy_samples = []
        while game_continues:
            root = mcts.run(game.state)
            action_dict = root.action_probs()
            action_probs = torch.zeros(game.num_actions(), dtype=torch.float32)
            for action, prob in action_dict.items():
                action_probs[action] = prob
            policy_samples.append((game.state.clone(), action_probs))
            # action = root.best_action()
            action = root.sample_action() # sample action according to visit counts
            game_continues = game.play(action)
            if not game_continues:
                reward = game.reward(game.state)
                winner = 1 - game.state[-1] # last element indicates losing player
                for state, action_probs in policy_samples:
                    # reward is always -1 since game ends in loser's turn else 0 for draw
                    r = -reward if winner == (state[-1]) else reward
                    r = torch.tensor(r, dtype=torch.float32)
                    samples.append((state, action_probs, r))
                break
    return samples

def create_dataloader(samples, batch_size=64):
    dataset = TensorDataset(
        torch.stack([s[0] for s in samples]),
        torch.stack([s[1] for s in samples]),
        torch.stack([s[2] for s in samples])
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def train_model(model, dataloader, optimizer, num_epochs=50):
    for epoch in range(num_epochs):
        losses = []
        for states, action_probs, rewards in dataloader:
            pred_probs, pred_value = model(states)
            value_loss = torch.nn.functional.mse_loss(pred_value.squeeze(), rewards)
            policy_loss = -torch.mean(torch.sum(action_probs * torch.log(pred_probs + 1e-8), dim=1))
            # regularization_loss = 1e-4 * sum(torch.sum(param ** 2) for param in model.parameters())
            loss = value_loss + policy_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        avg_loss = sum(losses) / len(losses)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")