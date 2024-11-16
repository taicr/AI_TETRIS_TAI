# Agent.py

import torch
import torch.nn as nn
from dqn_model import DQN

class Agent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(height=20, width=10, outputs=6).to(self.device)
        self.model.load_state_dict(torch.load('tetris_dqn.pth', map_location=self.device))
        self.model.eval()

    def choose_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
            action = torch.argmax(q_values).item()
        return action