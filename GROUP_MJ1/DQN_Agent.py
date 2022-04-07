import numpy as np
import torch
from torch import nn
from collections import deque, namedtuple
import random
from torch.utils.data import TensorDataset, DataLoader

class Agent():
    def __init__(self, env_specs):
        self.env_specs = env_specs
        self.device='cpu'
        self.buffer_size = 10000
        self.batch_size = 1000
        self.discount = 0.999
        self.eval_freq = 100
        
        self.memorybuffer = deque(maxlen=self.buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

        self.model = QNet()
        self.targetmodel = QNet()
        self.targetmodel.load_state_dict(self.model.state_dict())
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=0.001)
        self.loss = torch.nn.MSELoss()
        
    def put_in_buffer(self, state, action, reward, next_state, done):
        self.experience_tuple = self.experience(state, action, reward, next_state, done)
        self.memorybuffer.append(self.experience_tuple)
    
    def update(self, curr_obs, action, reward, next_obs, done, timestep):
        if curr_obs is not None:
            curr_obs = torch.tensor(curr_obs).permute(2, 0, 1)[None , ...].float().to(self.device)
            next_obs = torch.tensor(next_obs).permute(2, 0, 1)[None , ...].float().to(self.device)
            reward = torch.tensor([reward]).to(self.device)
            action = torch.tensor([action]).to(self.device)
            self.put_in_buffer(curr_obs, action, reward, next_obs, done)
        if len(self.memorybuffer)>= self.batch_size:
            #sample from memory buffer
            memory = random.sample(self.memorybuffer, k=self.batch_size)
            
            #put state, next state, rewards, and actions into tensors
            states = torch.cat([exp.state for exp in memory if exp is not None]).float().to(self.device)
            next_states = torch.cat([exp.next_state for exp in memory if exp is not None]).float().to(self.device)
            rewards = torch.cat([exp.reward for exp in memory if exp is not None]).float().to(self.device)
            actions = torch.cat([exp.action for exp in memory if exp is not None]).float().to(self.device)


            state_values = self.model(states).gather(1, actions.type(torch.int64).view(-1, 1))
            max_qs = self.targetmodel(next_states).max(1)[0].detach()
            
            expected_qs = (max_qs * self.discount) + rewards.view(-1, 1)

            #print(y.squeeze(1))
            loss = self.loss(state_values, expected_qs.squeeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            for param in self.model.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()
            
        if(timestep % self.eval_freq == 0 or done):
            self.targetmodel.load_state_dict(self.model.state_dict())
            self.save()
            
    def act(self, curr_obs, mode='eval'):
        if curr_obs is None:
            action = np.random.choice([0,1,2,3])
        else:
            obs = torch.tensor(curr_obs).permute(2, 0, 1)[None , ...].float().to(self.device)            
            action = self.model(obs).max(1)[1].item()
        return action
        
    def save(self):
        path = './'+'DQN.pt'
        torch.save(self.model.state_dict(),path )
    
class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.vizmodel = nn.Sequential(*[nn.Conv2d(in_channels=3, out_channels= 16,kernel_size= 1 ),
                        torch.nn.ReLU(),
                        nn.Conv2d(in_channels=16, out_channels= 16,kernel_size= (3,3) ), 
                        torch.nn.ReLU(),
                        nn.Conv2d(in_channels=16, out_channels= 16,kernel_size= (3,3) ), 
                        torch.nn.ReLU(),
                        nn.Flatten(), 
                        nn.Linear(11*11*16, 24), 
                        nn.Linear(24, 4)])

    def forward(self, x):
        return self.vizmodel(x)