from os import times
import numpy as np
import torch
from torch import nn
import torch
import sys
from collections import deque

class Agent():
  '''
  red 1 
  yellow 0.1
  green -.1
  blue -1
  '''

  def __init__(self, env_specs, use_bin=True, use_scent=True,
               use_viz=True,multithread=0, devices='cpu' ,
               batch_size=5, gamma= 0.9, store_model= '', store_freq=100):
    self.env_specs = env_specs
    
    #model parameters
    self.device=devices
    self.use_bin=use_bin
    self.use_scent=use_scent
    self.use_viz=use_viz
    self.model = ActorCritic(use_bin=self.use_bin, use_scent=self.use_scent, use_viz=self.use_viz).float()
    #parallel computation across multiple gpus 
    self.multithread=multithread
    
    if(self.multithread):
      self.model= torch.nn.DataParallel(self.model)
    self.model= self.model.to(self.device)
    self.optim = torch.optim.Adam(self.model.parameters(),lr=0.01)
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
    self.obs= deque([])
    
    self.upd=False

  def load_weights(self,location=''):
    if(len(location) >0):
      self.model.load_state_dict(torch.load(location,map_location=torch.device(self.devices)))


  def act(self, curr_obs, mode='eval'):
    '''
    0: forward
    1: left
    2: right
    3: stay still'''
    if (curr_obs is None or len(self.obs)<3):
      
      self.action=  self.env_specs['action_space'].sample()
      self.obs.append(curr_obs)
    
    else:
      self.upd=True
      self.obs.popleft() #remove last element
      self.obs.append(curr_obs) #add currrent observation
      
      #process input
      inpt = self.get_x()
      #generate prob dist
      _ , next_action =self.model(**inpt)
      if(mode=='train'):
        #sample
        action = np.random.choice([0,1,2,3], p=next_action.squeeze(0).detach().cpu().numpy())
        
        #store state 
        self.S_t.append(inpt)
        #store action 
        self.a.append(action)
      else:
        action =  np.random.choice([0,1,2,3], p=next_action.squeeze(0).detach().cpu().numpy())# next_action.squeeze(0).detach().cpu().numpy().argmax()
      self.action=action
    return self.action

  def update(self, curr_obs, action, reward, next_obs, done, timestep):
    
    
    if(curr_obs is not None and  self.upd ):
      #store reward
      self.R.append(reward)
      # update model model 
      
      if((timestep>0) and (timestep % self.batch_size == 0 or done ) and len(self.a)>3):
        
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
        
      
    if (done):
      self.upd=False
      self.obs=  deque([])
        


  def get_st_batch(self):
    '''Helper function to get model inputs as dict which is useful when updating the model'''
    out= {'x_viz':None, 'x_scent':None, 'x_bin':None}
    out_st={}
    out_sttp1= {}
    for k in out.keys():
      all_states = torch.stack([x[k] for x in self.S_t], dim=0).squeeze(dim=1)
      out_st[k] = all_states[:-1, ...]
      out_sttp1[k] = all_states[1:, ...]
    
    return out_st,out_st
  def get_x(self):
    '''helper function to get model inputs based on agent parameters'''
    out= {'x_viz':None, 'x_scent':None, 'x_bin':None}
    
    if(self.use_scent):
      out['x_scent']= torch.stack([torch.tensor(obs[0])[None,None,None, ...].float() for obs in self.obs], dim=1).to(self.device)
      
    if(self.use_viz):
      out['x_viz']= torch.stack([torch.tensor( obs[1]).permute(2, 0, 1)[None , ...].float() for obs in self.obs], dim=1).to(self.device)
      
    if(self.use_bin):
      out['x_bin']= torch.stack([torch.tensor(obs[2]).reshape(15,15,4).permute(2, 0, 1)[None , ...].float() for obs in self.obs],dim=1).to(self.device)
      
    return out

# Channel attention module 
class CAM(torch.nn.Module):
    def __init__(self, in_ch, ratio=2):
        super(CAM, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.max_pool = torch.nn.AdaptiveMaxPool2d(1)
           
        self.fc = torch.nn.Sequential(torch.nn.Conv2d(in_ch, in_ch // ratio, 1, bias=False),
                               torch.nn.ReLU(),
                               torch.nn.Conv2d(in_ch // ratio, in_ch, 1, bias=False))
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x*self.sigmoid(out)

# Spatial attention module 
class SAM(torch.nn.Module):
    def __init__(self, in_ch,relu_a=0.01):
        super().__init__()
        self.cnn_ops = [
            torch.nn.Conv2d(in_channels=2, out_channels=1, \
                            kernel_size=7, padding='same'),
            torch.nn.Sigmoid(), ] # use Sigmoid to norm to [0, 1]
        
        self.attention_layer = torch.nn.Sequential(*self.cnn_ops)
        
    def forward(self, x, ret_att=False):
        _max_out, _ = torch.max(x, 1, keepdim=True)
        _avg_out    = torch.mean(x, 1, keepdim=True)
        _out = torch.cat((_max_out, _avg_out), dim=1)
        _attention = _out
        for layer in self.attention_layer:
            _attention = layer(_attention)
           
        if ret_att:
            return _attention, _attention * x
        else:
            return _attention * x
        
class TimeDistributed(torch.nn.Module):
    def __init__(self, module, batch_first=True):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1,x.size(-3), x.size(-2),x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-3),y.size(-2),y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y
      
class ConvGRUCell(torch.nn.Module):
    
    def __init__(self,input_size,hidden_size,kernel_size):
        super(ConvGRUCell,self).__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.ConvGates   = torch.nn.Conv2d(self.input_size + self.hidden_size,2 * self.hidden_size,3,padding='same')
        self.Conv_ct     = torch.nn.Conv2d(self.input_size + self.hidden_size,self.hidden_size,3,padding='same') 
        dtype            = torch.FloatTensor
    
    def forward(self,input,hidden):
        if hidden is None:
           size_h    = [input.data.size()[0],self.hidden_size] + list(input.data.size()[2:])
           hidden    = torch.autograd.Variable(torch.zeros(size_h)).to(input.device)
        c1           = self.ConvGates(torch.cat((input,hidden),1))
        (rt,ut)      = c1.chunk(2, 1)
        reset_gate   = torch.sigmoid(rt)
        update_gate  = torch.sigmoid(ut)
        gated_hidden = torch.mul(reset_gate,hidden)
        p1           = self.Conv_ct(torch.cat((input,gated_hidden),1))
        ct           = torch.tanh(p1)
        next_h       = torch.mul(update_gate,hidden) + (1-update_gate)*ct
        return next_h
      
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
      # extract feattures
      self.viz_prep = TimeDistributed(nn.Sequential(*[nn.Conv2d(in_channels=3, out_channels= 8,kernel_size= 1 ),
                                       torch.nn.ReLU(),
                                       nn.Conv2d(in_channels=8, out_channels= 8,kernel_size= 1 ,),
                                       torch.nn.ReLU(),
                                       nn.Conv2d(in_channels=8, out_channels= 3,kernel_size= 1 ,)]))
      self.viz_grus = nn.ModuleList([ConvGRUCell(input_size=3,hidden_size=3,kernel_size = 3) for _ in range(3)])
      # shared cnn for viz feature extraction 
      self.viz_fwd = nn.Sequential(*[nn.Conv2d(in_channels=3, out_channels= 16,kernel_size= 1 ),
                                            torch.nn.ReLU(),
                                            nn.Conv2d(in_channels=16, out_channels= 16,kernel_size= (3,3) ), 
                                            torch.nn.ReLU(),
                                            nn.Conv2d(in_channels=16, out_channels= 16,kernel_size= (3,3) ), 
                                            torch.nn.ReLU(),
                                            nn.Flatten(), 
                                            nn.Linear(11*11*16, 24), 
                                            nn.Linear(24, 4)
      ])
      # critic extension
      self.critic_viz_fwd = nn.Sequential(*[ nn.ReLU(),
                                            nn.Linear(4, 1)])
      # actor extension
      self.actor_viz_fwd = nn.Sequential(*[nn.ReLU()])
      
    if self.use_scent:
      # extract feattures
      self.scent_prep = (nn.Sequential(*[TimeDistributed(nn.Conv2d(in_channels=1, out_channels= 8,kernel_size= 1 )),
                                       torch.nn.ReLU(),
                                       TimeDistributed(nn.Conv2d(in_channels=8, out_channels= 1,kernel_size= 1 ,))]))
      self.scent_grus = nn.ModuleList([ConvGRUCell(input_size=1,hidden_size=1,kernel_size = 1) for i in range(3)])
      # shared mlp for scent feature extraction 
      self.scent_fwd = nn.Sequential(*[nn.Flatten(),
                                              nn.Linear(3, 32),
                                              nn.ReLU(),
                                              nn.Linear(32, 64),
                                              nn.ReLU(),
                                              nn.Linear(64, 32),
                                              nn.ReLU()])
      # crritic extension
      self.critic_scent_fwd= nn.Sequential(*[ nn.Linear(32, 1)])
      # actor extension
      self.actor_scent_fwd = nn.Sequential(*[nn.Linear(32, 4)])
      
    if self.use_bin:
      # extract feattures
      self.bin_prep = TimeDistributed(nn.Sequential(*[nn.Conv2d(in_channels=4, out_channels= 8,kernel_size= 1 ),
                                       torch.nn.ReLU(),
                                       nn.Conv2d(in_channels=8, out_channels= 8,kernel_size= 1 ,),
                                       torch.nn.ReLU(),
                                       nn.Conv2d(in_channels=8, out_channels= 4,kernel_size= 1 ,)]))
      self.bin_grus = nn.ModuleList([ConvGRUCell(input_size=4,hidden_size=4,kernel_size = 3) for _ in range(3)])
      
      # shared cnn for binary encoding feature extraction 
      self.bin_fwd = nn.Sequential(*[nn.Conv2d(in_channels=4, out_channels= 16,kernel_size= 1 ),
                                            torch.nn.ReLU(),
                                            nn.Conv2d(in_channels=16, out_channels= 16,kernel_size= (3,3) ), 
                                            torch.nn.ReLU(),
                                            nn.Conv2d(in_channels=16, out_channels= 16,kernel_size= (3,3) ), 
                                            nn.Flatten(), 
                                            nn.Linear(11*11*16, 24)])
      #critic extension
      self.critic_bin_fwd = nn.Sequential(*[nn.Linear(24, 4), 
                                            nn.ReLU(),
                                            nn.Linear(4, 1)])
      #actor extension
      self.actor_bin_fwd = nn.Sequential(*[nn.Linear(24, 4)])
    #collating together extracted features from previous layers
    self.critic_out = nn.Sequential(*[nn.Flatten(), 
                                      nn.Linear(self.num_vars, 1)
                                      ])
    
    self.actor_out = nn.Sequential(*[nn.Flatten(), 
                                     nn.Linear(self.num_vars*4, 4),
                                     nn.Softmax(dim=1)])
      
      
  def forward(self, x_viz=None, x_scent=None, x_bin=None):
    #(BS,T, 3, 15, 15),
    #(BS,T,1, 1, 3),
    #(BS,T, 4, 15, 15)
    crit_out=[]
    act_out= []
    if self.use_viz:
      #extract features 
      viz= self.viz_prep(x_viz)
      #run rnn
      hidden_next=None
      for i in range(3):
        hidden_next = self.viz_grus[i](viz[:,i,...],hidden_next)
      viz= hidden_next
      viz = self.viz_fwd(viz)
      act_out.append(self.actor_viz_fwd(viz))
      crit_out.append(self.critic_viz_fwd(viz))
    if self.use_scent:
      scent = self.scent_prep(x_scent)
      #run rnn
      hidden_next=None
      for i in range(3):
        hidden_next = self.scent_grus[i](scent[:,i,...],hidden_next)
      scent= hidden_next
      scent = self.scent_fwd(scent)
      act_out.append(self.actor_scent_fwd(scent))
      crit_out.append(self.critic_scent_fwd(scent))
    if self.use_bin:
      bin = self.bin_prep(x_bin)
      #run rnn
      hidden_next=None
      for i in range(3):
        hidden_next = self.bin_grus[i](bin[:,i,...],hidden_next)
      bin= hidden_next
      bin = self.bin_fwd(bin)
      act_out.append(self.actor_bin_fwd(bin))
      crit_out.append(self.critic_bin_fwd(bin))
 
      
    crit_out = torch.stack(crit_out, dim=1)
    act_out = torch.stack(act_out, dim=1) 
    
    return self.critic_out(crit_out), self.actor_out(act_out)
