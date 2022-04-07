from os import times
import numpy as np
import torch
from torch import nn
import torch
import sys

class Agent():
  '''
  red 1 
  yellow 0.1
  green -.1
  blue -1
  '''

  def __init__(self, env_specs, use_bin=True, use_scent=True,
               use_viz=True,multithread=0, devices='cpu' ,
               batch_size=5, gamma= 0.9, store_model= '', store_freq=100,mdl_load=''):
    self.env_specs = env_specs
    
    #model parameters
    self.device=devices
    self.use_bin=use_bin
    self.use_scent=use_scent
    self.use_viz=use_viz
    self.model = ActorCritic(use_bin=self.use_bin, use_scent=self.use_scent, use_viz=self.use_viz).float()
    
    if(len(mdl_load)>0):
      self.model.load_state_dict(torch.load(mdl_load,map_location=torch.device(devices)))
    #parallel computation across multiple gpus 
    self.multithread=multithread
    
    if(self.multithread):
      self.model= torch.nn.DataParallel(self.model)
    self.model= self.model.to(self.device)
    self.optim = torch.optim.Adam(self.model.parameters(),lr=0.0001)
    self.critic_loss = torch.nn.MSELoss() # torch.nn.L1Loss()
    
    #model storing parameters
    self.store_model = store_model #location
    self.store_freq = store_freq #freq
    self.num_updates= 0
    
    #init action
    self.action=self.env_specs['action_space'].sample()
    
    # AC storing stack 
    self.batch_size=batch_size
    self.R= []
    self.prob_a= []
    self.S_t=[]
    self.a= []
    self.gam= gamma

  def load_weights(self,location=''):
    if(len(location) >0):
      self.model = self.model.load_state_dict(torch.load(location,map_location=torch.device(self.devices)))
      
    pass

  def act(self, curr_obs, mode='eval'):
    '''
    0: forward
    1: left
    2: right
    3: stay still'''
    if (curr_obs is None):
      self.action=  self.env_specs['action_space'].sample()
    
    else:
      #process input
      inpt =  self.get_x(curr_obs)
      #generate prob dist
      _ , next_action =self.model(**inpt)
      #sample
      action = np.random.choice([0,1,2,3], p=next_action.squeeze(0).detach().cpu().numpy())
      self.action=action
      #store state 
      if(mode=='train'):
        self.S_t.append(inpt)
        #store action 
        self.a.append(self.action)
    return self.action

  def update(self, curr_obs, action, reward, next_obs, done, timestep):
    
    if(curr_obs is not None ):
      #store reward
      self.R.append(reward)
    
      # update model model 
      if(timestep % self.batch_size == 0 or done ):
        
        #count number of updates 
        self.num_updates =self.num_updates+1
        
        #conversion of stored lists to tensors
        prev_states,next_states = self.get_st_batch() 
        prev_actions= torch.tensor(self.a[:-1]).reshape(len(self.a)-1, -1).to(self.device) #select all but last action as it has no st+1 associated with it 
        rewards= torch.tensor(self.R[:-1]).reshape(len(self.R)-1, -1).to(self.device) #select all but last reward as it has no st+1 associated with it 

        
        # current state value estim and policy sample
        v_st , pi_st = self.model(**prev_states)
        
        #next state value estim
        v_stp1, _ = self.model(**next_states)
        
        #td target
        td_target= rewards + self.gam * v_stp1
        
        #td error
        delta = td_target -v_st
        
        #logprob of actions taken
        log_probs = pi_st.gather(1,prev_actions ).log()
        
        #update model
        self.model.zero_grad()
        loss = -log_probs * delta.detach() + self.critic_loss(td_target.detach(), v_st) #add entropy for expl?
        loss.mean().backward()
        self.optim.step()   
        
        #store model 
        if(done or (len(self.store_model)>0 and (self.num_updates% self.store_freq)==0 )):
        
          if(self.multithread):
            torch.save(self.model.module.state_dict(), f"{self.store_model}/{self.store_model}-mdl-it{self.num_updates}.pth")
          else:
            torch.save(self.model.state_dict(), f"{self.store_model}/{self.store_model}-mdl-it{self.num_updates}.pth")
        
        # print some vals 
        print(f'cumulative rewards = {sum(self.R)} at timestep {timestep}')
        #print(self.a)
        #reset values 
        self.R= []
        self.S_t=[]
        self.a= []


  def get_st_batch(self):
    '''Helper function to get model inputs as dict which is useful when updating the model'''
    out= {'x_viz':None, 'x_scent':None, 'x_bin':None}
    out_st={}
    out_sttp1= {}
    for k in out.keys():
      all_states = torch.stack([x[k] for x in self.S_t], dim=0).squeeze()
      out_st[k] = all_states[:-1, ...]
      out_sttp1[k] = all_states[1:, ...]
    
    return out_st,out_st
  def get_x(self, obs):
    '''helper function to get model inputs based on agent parameters'''
    out= {'x_viz':None, 'x_scent':None, 'x_bin':None}
    
    if(self.use_scent):
      out['x_scent']= torch.tensor(obs[0])[None,None, ...].float().to(self.device)
      
    if(self.use_viz):
      out['x_viz']= torch.tensor(obs[1]).permute(2, 0, 1)[None , ...].float().to(self.device)
      
    if(self.use_bin):
      out['x_bin']= torch.tensor(obs[2]).reshape(15,15,4).permute(2, 0, 1)[None , ...].float().to(self.device)
      
    return out

class ActorCritic(nn.Module):
  '''This is a class of ActorCritic that takes in a
  visual representtation of the space and outputs
  a value associated with the state-action pair and also action to take.
  This critic can make use of any of the outputs of the environment'''
  def __init__(self, use_viz= True, use_scent= True, use_bin=True ):
    super().__init__()
    
    self.use_viz = use_viz
    self.use_scent = use_scent
    self.use_bin = use_bin
    
    self.num_vars=int(sum([use_viz,use_scent,use_bin]))
    
    if self.use_viz:
      #cnn for viz feature extraction 
      self.critic_viz_fwd = nn.Sequential(*[nn.Conv2d(in_channels=3, out_channels= 16,kernel_size= 1 ),
                                            torch.nn.ReLU(),
                                            nn.Conv2d(in_channels=16, out_channels= 16,kernel_size= (3,3) ), 
                                            torch.nn.ReLU(),
                                            nn.Conv2d(in_channels=16, out_channels= 16,kernel_size= (3,3) ), 
                                            torch.nn.ReLU(),
                                            nn.Flatten(), 
                                            nn.Linear(11*11*16, 24), 
                                            nn.Linear(24, 4), 
                                            nn.ReLU(),
                                            nn.Linear(4, 1)])
      
      self.actor_viz_fwd = nn.Sequential(*[nn.Conv2d(in_channels=3, out_channels= 16,kernel_size= 1 ),
                                           torch.nn.ReLU(),
                                           nn.Conv2d(in_channels=16, out_channels= 16,kernel_size= (3,3) ), 
                                           torch.nn.ReLU(),
                                           nn.Conv2d(in_channels=16, out_channels= 16,kernel_size= (3,3) ), 
                                           torch.nn.ReLU(),
                                           nn.Flatten(), 
                                           nn.Linear(11*11*16, 24), 
                                           nn.Linear(24, 4)])
      
    if self.use_scent:
      #mlp for scent feature extraction 
      self.critic_scent_fwd = nn.Sequential(*[nn.Flatten(),
                                              nn.Linear(3, 32),
                                              nn.ReLU(),
                                              nn.Linear(32, 64),
                                              nn.ReLU(),
                                              nn.Linear(64, 32),
                                              nn.ReLU(),
                                              nn.Linear(32, 1)])
      
      self.actor_scent_fwd = nn.Sequential(*[nn.Flatten(),
                                             nn.Linear(3, 32),
                                             nn.ReLU(),
                                             nn.Linear(32, 64),
                                             nn.ReLU(),
                                             nn.Linear(64, 32),
                                             nn.ReLU(),
                                             nn.Linear(32, 4)])
    if self.use_bin:
      #cnn for binary encoding feature extraction 
      self.critic_bin_fwd = nn.Sequential(*[nn.Conv2d(in_channels=4, out_channels= 16,kernel_size= 1 ),
                                            torch.nn.ReLU(),
                                            nn.Conv2d(in_channels=16, out_channels= 16,kernel_size= (3,3) ), 
                                            torch.nn.ReLU(),
                                            nn.Conv2d(in_channels=16, out_channels= 16,kernel_size= (3,3) ), 
                                            nn.Flatten(), 
                                            nn.Linear(11*11*16, 24), 
                                            nn.Linear(24, 4), 
                                            nn.ReLU(),
                                            nn.Linear(4, 1)])
      
      self.actor_bin_fwd = nn.Sequential(*[nn.Conv2d(in_channels=4, out_channels= 16,kernel_size= 1 ),
                                           torch.nn.ReLU(),
                                           nn.Conv2d(in_channels=16, out_channels= 16,kernel_size= (3,3) ),
                                           torch.nn.ReLU(), 
                                           nn.Conv2d(in_channels=16, out_channels= 16,kernel_size= (3,3) ), 
                                           torch.nn.ReLU(),
                                           nn.Flatten(), 
                                           nn.Linear(11*11*16, 24), 
                                           nn.Linear(24, 4)])
    #collating together extracted features from previous layers
    self.critic_out = nn.Sequential(*[nn.Flatten(), 
                                      nn.Linear(self.num_vars, 1)
                                      ])
    
    self.actor_out = nn.Sequential(*[nn.Flatten(), 
                                     nn.Linear(self.num_vars*4, 4),
                                     nn.Softmax(dim=1)])
      
      
  def forward(self, x_viz=None, x_scent=None, x_bin=None):
    crit_out=[]
    act_out= []
    if self.use_viz:
      adv_viz = self.actor_viz_fwd(x_viz)
      act_out.append(adv_viz)
      crit_viz= self.critic_viz_fwd(x_viz)
      crit_out.append(crit_viz)
      
    if self.use_scent:
      adv_scent = self.actor_scent_fwd(x_scent)
      act_out.append(adv_scent)
      crit_scent= self.critic_scent_fwd(x_scent)
      crit_out.append(crit_scent)
      
    if self.use_bin:
      adv_bin = self.actor_bin_fwd(x_bin)
      act_out.append(adv_bin)
      crit_bin= self.critic_bin_fwd(x_bin)
      crit_out.append(crit_bin)
      
    crit_out = torch.stack(crit_out, dim=1)
    act_out = torch.stack(act_out, dim=1) 
    
    return self.critic_out(crit_out), self.actor_out(act_out)