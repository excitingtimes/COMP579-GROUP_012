import numpy as np
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.optim import Adam
from torch import Tensor
class Agent():
  '''The agent class that is to be filled.
     You are allowed to add any method you
     want to this class.
     red 1 
  yellow 0.1
  green -.1
  blue -1
  '''

  def __init__(self, env_specs):
    self.env_specs = env_specs
    self.model = ReinforceViz()
    self.loss=0
    self.optim = Adam(self.model.parameters(),lr=0.1)
    self.action=self.env_specs['action_space'].sample()
    self.gt=0
    self.gam= 0.1

  def load_weights(self):
    pass

  def act(self, curr_obs, mode='eval'):
    '''
    0: forward
    1: left
    2: right
    3: stay still -> should be ignored'''
    return self.action
    #return 1#self.env_specs['action_space'].sample()

  def update(self, curr_obs, action, reward, next_obs, done, timestep):
    
    if curr_obs is not None:
      
      #update model 
      self.model.zero_grad()
      self.gt= self.gt + self.gam*reward
      _ , log_prob= self.model(Tensor(curr_obs[1]).permute(2, 0, 1)[None , ...])

      loss = (- self.gam ** timestep * self.gt * log_prob.squeeze(0)[action] )
      loss.backward()
      self.optim.step()

      #next action 
      next_action , _ =self.model(Tensor(next_obs[1]).permute(2, 0, 1)[None , ...])
      print(next_action)
      self.action = np.random.choice([0,1,2,3], p=next_action.squeeze(0).detach().numpy())
      
      print(self.action)
      print(next_action)


class ReinforceViz(nn.Module):
    def __init__(self):
        super(ReinforceViz,self).__init__()
        
        # 15 x 15 x 3 image input
        self.fwd = nn.Sequential(*[nn.Conv2d(in_channels=3, out_channels= 16,kernel_size= 1 ),
                                   nn.Conv2d(in_channels=16, out_channels= 16,kernel_size= (3,3) ), 
                                   nn.Conv2d(in_channels=16, out_channels= 16,kernel_size= (3,3) ), 
                                   nn.Flatten(), 
                                   nn.Linear(11*11*16, 24), 
                                   nn.Linear(24, 4), 
                                   nn.Softmax()])
        
                
    def forward(self,x):

      out= self.fwd(x)
      
      return out, out.log()
