from audioop import maxpp
import numpy as np
import torch

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
    self.target = {'set':False}
  def load_weights(self):
    pass

  def act(self, curr_obs, mode='eval'):
    '''
    0: forward
    1: left
    2: right
    3: stay still -> should be ignored''' 

    if self.target['set']:
      if(len(self.target['actions'])>0):
        action = self.target['actions'][0]
        self.target['actions'] = self.target['actions'][1:]
      else:
         self.target['set']= False
         action= self.env_specs['action_space'].sample()
    else:
      action= 3#self.env_specs['action_space'].sample()
    return action

  def update(self, curr_obs, action, reward, next_obs, done, timestep):
    
    if self.target['set']:
      #change the target path
      pass
    else:
      tmp = next_obs[2].reshape(15,15,4)
      target= tmp[:,:,0]+ 0.1*tmp[:,:,1]- 0.1*tmp[:,:,2]-tmp[:,:,3]
      max_reward = np.unravel_index(target.argmax(), target.shape)
      left = 7 - max_reward[0]
      right = max_reward[0] - 7
      down= 7-max_reward[1]
      up= max_reward[1] -7 
      
      actions =  [0 for _ in range(up)] +[1 for _ in range(left)] +  [2 for _ in range(right)] + [3 for _ in range(down)]
        
      self.target['actions'] = actions
      self.target['set']= True
      
      



