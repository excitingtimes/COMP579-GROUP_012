import warnings
# warnings.filterwarnings("ignore")
import jbw
import gym
from gym import Wrapper
from gym.wrappers import LazyFrames
from gym.spaces import Box

from collections import deque
import numpy as np
from tqdm import tqdm

from argparse import ArgumentParser
import warnings
warnings.filterwarnings("ignore")

import torch
from torchvision import transforms as T



class FrameStack(Wrapper):
    """
    Observation wrapper that stacks the observations in a rolling manner.
    For example, if the number of stacks is 4, then the returned observation contains
    the most recent 4 observations. For environment 'Pendulum-v0', the original observation
    is an array with shape [3], so if we stack 4 observations, the processed observation
    has shape [4, 3].
    note::
        To be memory efficient, the stacked observations are wrapped by :class:`LazyFrame`.
    note::
        The observation space must be `Box` type. If one uses `Dict`
        as observation space, it should apply `FlattenDictWrapper` at first.
    Example::
        >>> import gym
        >>> env = gym.make('PongNoFrameskip-v0')
        >>> env = FrameStack(env, 4)
        >>> env.observation_space
        Box(4, 210, 160, 3)
    Args:
        env (Env): environment object
        num_stack (int): number of stacks
        lz4_compress (bool): use lz4 to compress the frames internally
    """
    def __init__(self, env, num_stack, lz4_compress=False):
        super(FrameStack, self).__init__(env)
        self.num_stack = num_stack
        self.lz4_compress = lz4_compress

        self.vision_frames = deque(maxlen=num_stack)
        self.scent_frames = deque(maxlen=num_stack)
        self.feature_frames = deque(maxlen=num_stack)

        low = np.repeat(self.vision_space.low[np.newaxis, ...], num_stack, axis=0)
        high = np.repeat(self.vision_space.high[np.newaxis, ...], num_stack, axis=0)
        self.vision_space = Box(low=low, high=high, dtype=self.vision_space.dtype)
        
        scent_low = np.repeat(self.scent_space.low[np.newaxis, ...], num_stack, axis=0)
        scent_high = np.repeat(self.scent_space.high[np.newaxis, ...], num_stack, axis=0)
        self.scent_space = Box(low=scent_low, high=scent_high, dtype=self.scent_space.dtype)

        feature_low = np.repeat(self.feature_space.low[np.newaxis, ...], num_stack, axis=0)
        feature_high = np.repeat(self.feature_space.high[np.newaxis, ...], num_stack, axis=0)
        self.feature_space = Box(low=feature_low, high=feature_high, dtype=self.feature_space.dtype)

    def _get_vision(self):
        assert len(self.vision_frames) == self.num_stack, (len(self.vision_frames), self.num_stack)
        return LazyFrames(list(self.vision_frames), self.lz4_compress)
    
    def _get_scent(self):
        assert len(self.scent_frames) == self.num_stack, (len(self.scent_frames), self.num_stack)
        return LazyFrames(list(self.scent_frames), self.lz4_compress)

    def _get_feature(self):
        assert len(self.feature_frames) == self.num_stack, (len(self.feature_frames), self.num_stack)
        return LazyFrames(list(self.feature_frames), self.lz4_compress)

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        scent, observation, feature, moved = state
        self.vision_frames.append(observation)
        self.scent_frames.append(scent)
        self.feature_frames.append(feature)
        return ( self._get_scent(), self._get_vision(), self._get_feature(), moved), reward, done, info

    def reset(self, **kwargs):
        state = self.env.reset(**kwargs)
        scent, observation, feature, moved = state
        [self.vision_frames.append(observation) for _ in range(self.num_stack)]
        [self.scent_frames.append(scent) for _ in range(self.num_stack)]
        [self.feature_frames.append(feature) for _ in range(self.num_stack)]
        return (self._get_scent(), self._get_vision(), self._get_feature(), moved)

class ObservationWrapper(Wrapper):
    def reset(self, **kwargs):
        state = self.env.reset(**kwargs)
        scent, obs, feature, moved = state
        return (self.get_scent(scent), self.get_observation(obs), self.get_feature(feature), moved)

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        scent, obs, feature, moved = state
        return (self.get_scent(scent), self.get_observation(obs), self.get_feature(feature), moved), reward, done, info

    def permute_orientation(self, observation):
        # permute [H, W, C] array to [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def get_observation(self, observation):
        raise NotImplementedError

    def get_scent(self, scent):
        raise NotImplementedError

    def get_feature(self, feature):
        raise NotImplementedError

class GrayScaleObservation(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.vision_space.shape[:2]
        self.vision_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def get_observation(self, observation):
        if isinstance(observation, torch.Tensor):
            observation = observation.numpy()
        observation = self.permute_orientation(observation)
        transform = T.Compose([T.ToPILImage(), T.Grayscale(), T.ToTensor()])
        observation = transform(observation).squeeze()
        return observation

    def get_scent(self, scent):
        return scent

    def get_feature(self, feature):
        return feature



def gather_dataset(
    n_samples=10_000, 
    num_stack=10, 
    delta_stack=5, 
    max_samples_per_episode=1_000, 
    save_directory=None,
):
    n_gathered_samples = 0
    delta = 0
    i = 0

    _dim_scent_ = [num_stack, 3]
    _dim_vision_ = [num_stack, 16, 16, 3]
    _dim_feature_ = [num_stack, 900]
    dim_scent = [n_samples] + _dim_scent_
    dim_vision = [n_samples] + _dim_vision_
    dim_feature = [n_samples] + _dim_feature_
    dim_episode = [n_samples, 1]

    print("\nDim Scent : ", dim_scent)
    print("Dim Vision : ", dim_vision)
    print("Dim Feature : ", dim_feature)
    print("Dim Episode : ", dim_episode)

    gathered_scent = torch.zeros(dim_scent)
    gathered_vision = torch.zeros(dim_vision)
    gathered_feature = torch.zeros(dim_feature)
    gathered_episode = torch.zeros(dim_episode)

    logged_shapes = False
    with tqdm(total=n_gathered_samples) as pbar:
        env = gym.make("JBW-COMP579-obj-v1", render=False)
        stacked_env = FrameStack(env, num_stack=num_stack, lz4_compress=False)
        _ = stacked_env.reset()
        done = False
        while not done:
            svfm, reward, done, info  = stacked_env.step(stacked_env.action_space.sample())
            scent, vision, feature, moved = svfm
            delta += 1
            # Every {delta_stack} frames, we stack the {num_stack} frames together and add them to our dataset
            if (delta % delta_stack) == 0:
                pbar.set_description(f"Processing episode n°{str(i // max_samples_per_episode)} / {str(n_samples // max_samples_per_episode)} : {str(i)} samples collected in total")
                i += 1
                pbar.update(1)
                if logged_shapes == False:
                    logged_shapes = scent[:].shape, vision[:].shape, feature[:].shape
                gathered_scent[i] = torch.Tensor(scent[:])
                gathered_vision[i, :, :15, :15, :] = torch.Tensor(vision[:])
                gathered_feature[i] = torch.Tensor(feature[:])
                gathered_episode[i] = delta
                # For every {max_samples_per_episode} samples collected in our dataloader, we reset the environment to sample a new trajectory
                if (i % max_samples_per_episode) == 0:
                    _ = stacked_env.reset()

            if (i == n_samples - 1):
                print("Finished gathering samples")
                pbar.update(-1)
                break

    print("\nScent's shape : ", logged_shapes[0])
    print("Vision's shape : ", logged_shapes[1])
    print("Feature's shape : ", logged_shapes[2])

    if save_directory is not None:
        save_file = f"dataset-jellybean-num_stack={num_stack}.pt"
        print("Saving the dataset to disk ...")
        torch.save(
            obj={
                "scent": gathered_scent, 
                "vision": gathered_vision, 
                "feature": gathered_feature,
                "episode": gathered_episode,
            }, 
            f=save_directory+save_file,
        )
        print(f"Saved the dataset to <{save_directory + save_file}>\n")

        # out = torch.load(f=save_directory+save_file)
        
    return gathered_scent, gathered_vision, gathered_feature, gathered_episode



env = gym.make("JBW-COMP579-obj-v1", render=False)
stacked_env = FrameStack(env, num_stack=10, lz4_compress=False)
_ = stacked_env.reset()

for num_stack in [1, 2, 4, 8, 12, 16, 24, 32]:
    gather_dataset(
        num_stack=num_stack, 
        delta_stack=max(1, num_stack // 2),
        save_directory="/home/mila/s/sonnery.hugo/scratch/datasets/jellyean/",
    )