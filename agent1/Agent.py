import os
import torch
import torch.nn as nn
import numpy as np

# Sử dụng class DQN từ file huấn luyện
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(inplace=True))
        self.fc2 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(inplace=True))
        self.fc3 = nn.Linear(64, output_dim)

        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

class Agent:
    def __init__(self, turn):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        weight_file_path = os.path.join(dir_path, turn, 'weight')

        # Cấu hình mô hình
        self.input_dims = 20 * 17  # Kích thước đầu vào
        self.n_actions = 6  # Số lượng hành động từ mô hình huấn luyện
        self.network = DQN(self.input_dims, self.n_actions)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.network.to(self.device)

        # Load trọng số đã huấn luyện
        self.network.load_state_dict(torch.load(weight_file_path, map_location=self.device))
        self.network.eval()  # Chuyển sang chế độ đánh giá

    def preprocess_observation(self, observation):
        """
        Tiền xử lý quan sát:
        - Ghép lưới (20x10) và features (20x7) thành vector 20x17.
        - Flatten về 1 chiều.
        """
        grid = observation[:20, :10]  # Lưới 20x10
        features = observation[:20, 10:17]  # Feature vector 20x7
        combined = np.concatenate((grid, features), axis=1)  # Kết hợp thành 20x17
        return combined.flatten()  # Flatten thành vector 1D

    def choose_action(self, observation):
        """
        Chọn hành động dựa trên quan sát:
        - Xử lý observation để làm đầu vào mạng.
        - Dùng mạng để tính toán hành động.
        """
        # Tiền xử lý quan sát
        state = self.preprocess_observation(observation)
        state = torch.tensor(state, dtype=torch.float32).to(self.device)

        # Forward qua mạng để lấy Q-values
        with torch.no_grad():
            q_values = self.network(state.unsqueeze(0))  # Thêm batch dimension
        action = torch.argmax(q_values).item()  # Chọn hành động có Q-value cao nhất
        return action
