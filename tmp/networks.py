import os
import torch as t
import torch.nn.functional as f
import torch.nn as nn
import torch.optim as optim
#import torch.distributions.normal import Normal

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, fc1_dims=256, fc2_dims=128, name = 'critic', checkpoint_dir = 'tmp/ppo', learning_rate=10e-3):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_ppo')
        
        self.fc1 = nn.Linear(self.input_dims[0]+n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q1 = nn.Linear(self.fc2_dims, 1)
        
        #for optimizer
        self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=0.005)
        self.device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')
        print(f"Created Critic Network on device: {self.device},")
        
        self.to(self.device)
        
    def forward(self, state, action):
        action_value = self.fc1(t.cat([state, action], dim=1))
        action_value = f.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = f.relu(action_value)
        
        q1 = self.q1(action_value)
        return q1
    
    def save_checkpoint(self):
        t.save(self.state_dict(), self.checkpoint_file)
        
    def load_checkpoint(self):
        self.load_state_dict(t.load(self.checkpoint_file))
        
class ActorNetwork(nn.Module):
    def __init__(self, input_dims, fc1_dims=256, fc2_dims=128, learning_rate=10e-3, n_actions=2, name='actor', checkpoint_dir='tmp/ppo'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_ppo')

        self.fc1 = nn.Linear(self.input_dims[0], self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.output = nn.Linear(self.fc2_dims, self.n_actions)
        
        #for optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')
        print(f"Created Actor Network on device: {self.device},")
        
        self.to(self.device)
    
    def forward(self, state):
        x = self.fc1(state)
        x = f.relu(x)
        x = self.fc2(x)
        x = f.relu(x)
        
        x = t.tanh(self.output(x))
        
        return x
    
    def save_checkpoint(self):
        t.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        self.load_state_dict(t.load(self.checkpoint_file))