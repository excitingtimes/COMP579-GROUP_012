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
        self.buffer_size = 1000000
        self.batch_size = 512
        self.discount = 0.999
        self.eval_freq = 100
        
        self.memorybuffer = deque(maxlen=self.buffer_size)
        self.experience = namedtuple("Experience", field_names=[
                        "visualstate","scentstate","featurestate", 
                        "action", "reward",
                        "visualnext", "scentnext", "featurenext", "done"])

        self.model = QNet()
        self.targetmodel = QNet()
        self.targetmodel.load_state_dict(self.model.state_dict())
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=0.001)
        self.loss = nn.SmoothL1Loss()#torch.nn.MSELoss()
        
    def put_in_buffer(self, state0, state1, state2, action, reward, next_state0, next_state1, next_state2, done):
        self.experience_tuple = self.experience(state0, state1, state2, action, reward, next_state0, next_state1, next_state2, done)
        self.memorybuffer.append(self.experience_tuple)
    
    def update(self, curr_obs, action, reward, next_obs, done, timestep):
        if (curr_obs is not None) and (done is False):
            current_sight, current_scent, current_features = self.makeTensors(curr_obs)
            next_sight, next_scent, next_features = self.makeTensors(next_obs)

            reward = torch.tensor([reward]).to(self.device)
            action = torch.tensor([action]).to(self.device)
            self.put_in_buffer(current_sight, current_scent, current_features, action, reward, next_sight, next_scent, next_features, done)
        if len(self.memorybuffer)>= self.batch_size:
            #sample from memory buffer
            memory = random.sample(self.memorybuffer, k=self.batch_size)
            
            #put state, next state, rewards, and actions into tensors
            visual_states = torch.cat([exp.visualstate for exp in memory if exp is not None]).float().to(self.device)
            scent_states = torch.cat([exp.scentstate for exp in memory if exp is not None]).float().to(self.device)
            next_visual_states = torch.cat([exp.visualnext for exp in memory if exp is not None]).float().to(self.device)
            next_scent_states = torch.cat([exp.scentnext for exp in memory if exp is not None]).float().to(self.device)

            rewards = torch.cat([exp.reward for exp in memory if exp is not None]).float().to(self.device)
            actions = torch.cat([exp.action for exp in memory if exp is not None]).float().to(self.device)


            state_values = self.model(visual_states,scent_states).gather(1, actions.type(torch.int64).view(-1, 1))
            max_qs = self.targetmodel(next_visual_states,next_scent_states).max(1)[0].detach().view(-1, 1)
            expected_qs = (max_qs * self.discount) + rewards.view(-1, 1)
            loss = self.loss(state_values, expected_qs)#.squeeze(1))
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
            current_sight, current_scent, current_features = self.makeTensors(curr_obs)
            #obs = torch.tensor(curr_obs).permute(2, 0, 1)[None , ...].float().to(self.device)            
            action = self.model(current_sight,current_scent).max(1)[1].item()
        return action
        
    def save(self):
        path = './'+'DQN.pt'
        torch.save(self.model.state_dict(),path )
    
    def makeTensors(self, obs):
        smell = torch.tensor(obs[0]).to(self.device)[None,None, ...].float().to(self.device)
        sight = torch.tensor(obs[1]/255).permute(2, 0, 1)[None , ...].float().to(self.device)
        features = torch.tensor(obs[2]).reshape(15,15,4).permute(2, 0, 1)[None , ...].float().to(self.device)
        return sight, smell, features

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

        self.scentmodel = nn.Sequential(*[nn.Flatten(),
                                            nn.Linear(3, 32),
                                            nn.ReLU(),
                                            nn.Linear(32, 64),
                                            nn.ReLU(),
                                            nn.Linear(64, 32),
                                            nn.ReLU(),
                                            nn.Linear(32, 4)])

        self.linear = nn.Sequential(*[nn.ReLU(), 
                                    nn.Flatten(),
                                    nn.Linear(8, 16),
                                    nn.ReLU(),
                                    nn.Linear(16, 16),
                                    nn.ReLU(),
                                    nn.Linear(16, 4)])

    def forward(self, vision, scent):
        vision_cnn = self.vizmodel(vision)
        scent_mlp = self.scentmodel(scent)
        combined=torch.cat((vision_cnn, scent_mlp), dim=1)
        estimated_qs = self.linear(combined)
        return estimated_qs