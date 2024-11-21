import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from TetrisBattle.envs.tetris_env import TetrisDoubleEnv  

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()

        # Fully connected layers
        self.fc1 = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(inplace=True))
        self.fc2 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(inplace=True))
        self.fc3 = nn.Linear(64, output_dim)

        # Initialize weights
        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # Xavier initialization
                nn.init.constant_(m.bias, 0)       # Initialize biases to 0

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


# Hàm train mô hình DQN
def train():
    # Khởi tạo môi trường
    env = TetrisDoubleEnv(gridchoice="none", obs_type="grid", mode="rgb_array")
    input_dim = 20 * 17  # Kích thước đầu vào (20x17 vector đã flatten)
    num_actions = env.action_space.n  # Số lượng hành động

    # Tạo mô hình và target model
    model = DQN(input_dim, num_actions)
    target_model = DQN(input_dim, num_actions)
    target_model.load_state_dict(model.state_dict())  # Sao chép trọng số ban đầu từ model sang target_model
    target_model.eval()

    # Optimizer và Replay Buffer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    replay_buffer = deque(maxlen=10000)

    # Siêu tham số
    batch_size = 256
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.1
    target_update_freq = 10
    max_episodes = 100

    # Huấn luyện qua các tập
    for episode in range(max_episodes):
        state = env.reset()  # Reset môi trường
        state = preprocess_state(state)  # Xử lý đầu vào
        total_reward = 0
        done = False

        while not done:
            # Chọn hành động theo epsilon-greedy
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Thêm batch dimension
                    q_values = model(state_tensor)
                    action = torch.argmax(q_values).item()

            # Thực hiện hành động
            next_state, reward, done, _ = env.step(action)
            next_state = preprocess_state(next_state)

            # Lưu trải nghiệm vào replay buffer
            replay_buffer.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            # Huấn luyện từ replay buffer
            if len(replay_buffer) >= batch_size:
                train_from_replay_buffer(model, target_model, replay_buffer, optimizer, batch_size, gamma)

        # Giảm epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        # Cập nhật trọng số target model
        if episode % target_update_freq == 0:
            target_model.load_state_dict(model.state_dict())

        print(f"Episode {episode + 1}/{max_episodes}, Total Reward: {total_reward}")

    # Lưu trọng số của mô hình
    save_model(model, "weight.pth")

# Hàm xử lý trạng thái đầu vào
def preprocess_state(state):
    """
    Tiền xử lý state đầu vào:
    - Ghép bản đồ (20x10) và feature vector (20x7) thành vector 20x17
    - Flatten về 1 chiều
    """
    grid = state[:20, :10]  # Bản đồ kích thước (20x10)
    features = state[:20, :7]  # Feature vector (20x7)
    combined = np.concatenate((grid, features), axis=1)  # Kết hợp thành (20x17)
    return combined.flatten()  # Flatten về 1 chiều

# Hàm huấn luyện từ replay buffer
def train_from_replay_buffer(model, target_model, replay_buffer, optimizer, batch_size, gamma):
    minibatch = random.sample(replay_buffer, batch_size)
    states, actions, rewards, next_states, dones = zip(*minibatch)

    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)

    # Dự đoán Q-values cho state hiện tại
    q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    # Dự đoán Q-values cho state tiếp theo
    next_q_values = target_model(next_states).max(1)[0]
    # Tính target
    targets = rewards + gamma * next_q_values * (1 - dones)

    # Tính loss và tối ưu hóa
    loss = F.mse_loss(q_values, targets.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Hàm lưu mô hình
def save_model(model, filepath):
    directory = os.path.dirname(filepath)
    if directory != '':
        os.makedirs(directory, exist_ok=True)
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")

if __name__ == "__main__":
    train()