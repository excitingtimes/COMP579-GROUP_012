import gym
import jbw

import argparse
import importlib
import time
import random
import numpy as np
import tensorflow as tf
import torch
import argparse 
import os,shutil, sys
from os import listdir, makedirs
from os.path import isfile, join
import torch

from environments import JellyBeanEnv, MujocoEnv


def evaluate_agent(agent, env, n_episodes_to_evaluate):
  '''Evaluates the agent for a provided number of episodes.'''
  array_of_acc_rewards = []
  for _ in range(n_episodes_to_evaluate):
    acc_reward = 0
    done = False
    curr_obs = env.reset()
    while not done:
      action = agent.act(curr_obs, mode='eval')
      next_obs, reward, done, _ = env.step(action)
      acc_reward += reward
      curr_obs = next_obs
    array_of_acc_rewards.append(acc_reward)
  print('evaluated')
  return np.mean(np.array(array_of_acc_rewards))


def get_environment(env_type):
  '''Generates an environment specific to the agent type.'''
  if 'jellybean' in env_type:
    env = JellyBeanEnv(gym.make('JBW-COMP579-obj-v1',render=True))
  elif 'mujoco' in env_type:
    env = MujocoEnv(gym.make('Hopper-v2'))
  else:
    raise Exception("ERROR: Please define your agent_type to be either a 'JellyBeanAgent' or a 'MujocoAgent'!")
  return env


def train_agent(agent,
                env,
                total_timesteps,
                evaluation_freq,
                n_episodes_to_evaluate):

  seed = 0
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  tf.random.set_seed(seed)

  timestep = 0
  array_of_mean_acc_rewards = []
  
    
  while timestep < total_timesteps:
    
    done = False
    curr_obs = env.reset()
    while not done:    
      env.render(mode="matplotlib")
      #print(curr_obs)
      action = agent.act(curr_obs, mode='train')
      next_obs, reward, done, _ = env.step(action)
      agent.update(curr_obs, action, reward, next_obs, done, timestep)
      curr_obs = next_obs
      timestep += 1
      if timestep % evaluation_freq == 0:
        mean_acc_rewards = evaluate_agent(agent, env, n_episodes_to_evaluate)
        print('timestep: {ts}, acc_reward: {acr:.2f}'.format(ts=timestep, acr=mean_acc_rewards))
        array_of_mean_acc_rewards.append(mean_acc_rewards)
        curr_obs=env.reset()

  return array_of_mean_acc_rewards


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='')
  parser.add_argument('--group', type=str, default='GROUP_MJ1', help='group directory')
  parser.add_argument('-expName',type=str, default="debug", help='Experiment name')
  parser.add_argument('-use_bin',type=int, default=1, help='bool flag for binary')
  parser.add_argument('-use_scent',type=int, default=1, help='bool flag for scent vector')
  parser.add_argument('-use_viz',type=int, default=1, help='bool flag for viz')
  parser.add_argument('-batch_size', type=int, default=10, help='model batch size')
  parser.add_argument('-gamma',type=float, default=0.9, help='discount factor')
  parser.add_argument('-store_freq',type=int, default=100, help='numbrof updates until model is saved')
  parser.add_argument('-load_mdl',type=str, default='', help='path of model to load')
  args = parser.parse_args()
  path = './'+args.group+'/'
  files = [f for f in listdir(path) if isfile(join(path, f))]
  if ('agent.py' not in files) or ('env_info.txt' not in files):
    exit()

  with open(path+'env_info.txt') as f:
    lines = f.readlines()
  env_type = lines[0].lower()

  env = get_environment(env_type) 
  if 'jellybean' in env_type:
    env_specs = {'scent_space': env.scent_space, 'vision_space': env.vision_space, 'feature_space': env.feature_space, 'action_space': env.action_space}
  if 'mujoco' in env_type:
    env_specs = {'observation_space': env.observation_space, 'action_space': env.action_space}
  agent_module = importlib.import_module(args.group+'.actor_critic_agent')

  # Store print outputs to a file 
  itr_out_dir = args.expName + '-itrOut'
  if os.path.isdir(itr_out_dir): 
      shutil.rmtree(itr_out_dir)
  os.mkdir(itr_out_dir)
  sys.stdout = open(os.path.join(itr_out_dir, 'iter_prints.log'), 'w') 
  
  # prepare model env
  torch_devs = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  multithread = torch.cuda.device_count() > 1
  

  #define agent 
  agent = agent_module.Agent(env_specs,use_bin=args.use_bin, use_scent=args.use_scent,
                             use_viz=args.use_viz,multithread=multithread, devices=torch_devs,
                             batch_size=args.batch_size, gamma= args.gamma, store_model= itr_out_dir,
                             store_freq=args.store_freq,mdl_load=args.load_mdl)
  

  total_timesteps = 2000000
  evaluation_freq = 100
  n_episodes_to_evaluate = 20

  
  learning_curve = train_agent(agent, env, total_timesteps, evaluation_freq, n_episodes_to_evaluate)
  

