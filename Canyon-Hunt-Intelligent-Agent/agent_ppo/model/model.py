import torch
import torch.nn as nn
import numpy as np
from agent_ppo.conf.conf import Config

def make_fc_layer(in_features, out_features, gain=np.sqrt(2)):
    fc = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(fc.weight.data, gain=gain)
    nn.init.zeros_(fc.bias.data)
    return fc

class Model(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.model_name = "gorge_chase_lite"
        self.device = device

        input_dim = Config.DIM_OF_OBSERVATION
        action_num = Config.ACTION_NUM
        value_num = Config.VALUE_NUM
        
        self.map_size = 4 * 21 * 21
        self.state_dim = input_dim - self.map_size
        
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 21x21 -> 10x10
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 10x10 -> 5x5
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()     # 5x5 展开
        )
        
        cnn_out_dim = 128 * 5 * 5
        combined_dim = self.state_dim + cnn_out_dim
        
        # 显著扩大网络容量（大模型结构）
        hidden_dim = 1536
        mid_dim = 512
        final_dim = 256

        # PPO BEST PRACTICE 1: Separate Actor and Critic
        # 将网络加深为三层，并成倍增加神经元
        self.actor_net = nn.Sequential(
            make_fc_layer(combined_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            make_fc_layer(hidden_dim, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.ReLU(),
            make_fc_layer(mid_dim, final_dim),
            nn.LayerNorm(final_dim),
            nn.ReLU(),
        )
        
        self.critic_net = nn.Sequential(
            make_fc_layer(combined_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            make_fc_layer(hidden_dim, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.ReLU(),
            make_fc_layer(mid_dim, final_dim),
            nn.LayerNorm(final_dim),
            nn.ReLU(),
        )

        # PPO BEST PRACTICE 2: 0.01 Gain for Actor Head (uniform init dist)
        self.actor_head = make_fc_layer(final_dim, action_num, gain=0.01)
        self.critic_head = make_fc_layer(final_dim, value_num, gain=1.0)
        
        # PPO BEST PRACTICE 3: Orthogonal init on Conv2d
        for m in self.cnn.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight.data, gain=np.sqrt(2))
                nn.init.zeros_(m.bias.data)

    def forward(self, obs, inference=False):
        state_obs = obs[:, :self.state_dim]
        map_obs = obs[:, self.state_dim:]
        map_obs = map_obs.view(-1, 4, 21, 21)
        cnn_out = self.cnn(map_obs)
        
        combined = torch.cat([state_obs, cnn_out], dim=1)
        
        # Independent branches
        actor_hidden = self.actor_net(combined)
        critic_hidden = self.critic_net(combined)
        
        logits = self.actor_head(actor_hidden)
        value = self.critic_head(critic_hidden)
        
        return logits, value

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
