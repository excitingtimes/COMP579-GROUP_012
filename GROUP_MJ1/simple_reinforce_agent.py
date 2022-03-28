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
    self.optim = Adam(self.model.parameters(),lr=0.001)
    self.action=self.env_specs['action_space'].sample()
    self.gt=0
    self.gam= 0.1
    self.add_scent=True

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
      
      _ , log_prob= self.model(Tensor(curr_obs[1]).permute(2, 0, 1)[None , ...])
      
      #weight reward with scent
      reward= reward +  scent_from_fruits(next_obs[0])-scent_from_fruits(curr_obs[0]) if self.add_scent else reward
            
      self.gt= self.gt + self.gam*reward
      loss = (- self.gam ** timestep * self.gt * log_prob.squeeze(0)[action] )
      loss.backward()
      self.optim.step()

      #next action 
      next_action , _ =self.model(Tensor(next_obs[1]).permute(2, 0, 1)[None , ...])
      self.action = np.random.choice([0,1,2,3], p=next_action.squeeze(0).detach().numpy())
  
def nom_eucl(a,b):
  return np.linalg.norm(a/np.linalg.norm(a)- b/np.linalg.norm(b))

def scent_from_fruits(x):
  apple = np.array([1.64, 0.54, 0.4])
  banana = np.array([1.92, 1.76, 0.4])
  jelly = np.array([0.68, 0.01, 0.99])
  truffle = np.array([8.4, 4.8, 2.6])

  return nom_eucl(x, apple) + 0.1 * nom_eucl(x, banana) - nom_eucl(x, jelly) - nom_eucl(x,truffle)

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
