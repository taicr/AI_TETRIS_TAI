# train.py

import torch
import torch.optim as optim
import torch.nn.functional as F
from TetrisBattle import TetrisEnv
from dqn_model import DQN
import numpy as np
import random
from collections import deque

def train():
    env = TetrisEnv()
    num_actions = env.action_space.n
    height, width = env.observation_space.shape
    model = DQN(height, width, num_actions)
    target_model = DQN(height, width, num_actions)
    target_model.load_state_dict(model.state_dict())

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    memory = deque(maxlen=10000)
    batch_size = 64
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.1
    episodes = 1000
    target_update = 10
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    q_values = model(state_tensor)
                    action = torch.argmax(q_values).item()

            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            memory.append((state, action, reward, next_state, done))
            state = next_state

            if len(memory) >= batch_size:
                minibatch = random.sample(memory, batch_size)
                states_mb, actions_mb, rewards_mb, next_states_mb, dones_mb = zip(*minibatch)

                states_mb = torch.stack([torch.tensor(s, dtype=torch.float32) for s in states_mb]).unsqueeze(1)
                actions_mb = torch.tensor(actions_mb, dtype=torch.long)
                rewards_mb = torch.tensor(rewards_mb, dtype=torch.float32)
                next_states_mb = torch.stack([torch.tensor(s, dtype=torch.float32) for s in next_states_mb]).unsqueeze(1)
                dones_mb = torch.tensor(dones_mb, dtype=torch.float32)

                q_values = model(states_mb)
                q_values = q_values.gather(1, actions_mb.unsqueeze(1)).squeeze(1)

                next_q_values = target_model(next_states_mb).max(1)[0]
                expected_q_values = rewards_mb + gamma * next_q_values * (1 - dones_mb)

                loss = F.mse_loss(q_values, expected_q_values.detach())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        if episode % target_update == 0:
            target_model.load_state_dict(model.state_dict())

        print(f"Episode {episode}, Total Reward: {total_reward}")

    # Lưu mô hình
    torch.save(model.state_dict(), "tetris_dqn.pth")

if __name__ == "__main__":
    train()    