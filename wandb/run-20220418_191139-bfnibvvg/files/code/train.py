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

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import DeviceStatsMonitor
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.loggers import WandbLogger

import pl_bolts as pb
from pl_bolts.callbacks import PrintTableMetricsCallback
from pl_bolts.callbacks import BatchGradientVerificationCallback
from pl_bolts.callbacks import TrainingDataMonitor
from pl_bolts.models.autoencoders import AE

import torch
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader, random_split

from einops import rearrange

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



class JellybeanDataset(Dataset):
    def __init__(self, dataset_file):
        self.dataset_file = dataset_file
        data = torch.load(f=self.dataset_file)
        self.scent, self.vision, self.feature = data["scent"], data["vision"], data["feature"]

    def __len__(self):
        return len(self.scent)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {
            'scent': self.scent[idx], 
            'vision': rearrange(self.vision[idx], 'b num_stack h w c -> b (num_stack c) h w'),
            'feature': self.feature[idx],
        }

        return sample

class JellybeanDataModule(pl.LightningDataModule):
    def __init__(self, dataset_file="/home/mila/s/sonnery.hugo/scratch/datasets/jellyean/dataset-jellybean-num_stack=1.pt", batch_size=64):
        super().__init__()
        self.dataset_file = dataset_file
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.jellybean_full = JellybeanDataset(
            self.dataset_file,
        )
        self.jellybean_train, self.jellybean_val, self.jellybean_test = random_split(self.jellybean_full, [70_000, 20_000, 10_000])

    def train_dataloader(self):
        return DataLoader(
            self.jellybean_train, 
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.jellybean_val, 
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.jellybean_test, 
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )



def main(hparams):
    save_file = f"dataset-jellybean-num_stack={hparams.num_stack}.pt"
    save_directory = "/home/mila/s/sonnery.hugo/scratch/datasets/jellyean/"
    dataset_file = save_directory + save_file
    exp_name = f"autoencoder-num_stack={str(hparams.num_stack)}-mode={hparams.mode}"

    model_class = lambda num_stack: AE(
        input_height=15, 
        enc_type='resnet18', 
        first_conv=False, 
        maxpool1=False, 
        enc_out_dim=512, 
        latent_dim=256, 
        lr=0.0001,
    )

    # create your own theme!
    progress_bar_callback = RichProgressBar(
        theme=RichProgressBarTheme(
            description="green_yellow",
            progress_bar="green1",
            progress_bar_finished="green1",
            progress_bar_pulse="#6206E0",
            batch_progress="green_yellow",
            time="grey82",
            processing_speed="grey82",
            metrics="grey82",
        ),
    )
    print_table_metrics_callback = PrintTableMetricsCallback()
    seed_everything(42, workers=True)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=10,
        monitor="val_loss",
        mode="min",
        dirpath=hparams.default_root_dir,
        filename=exp_name + "-epoch={epoch:02d}-val_loss={val_loss:.2f}",
    )
    device_stats_monitor_callback = DeviceStatsMonitor()
    early_stop_callback = EarlyStopping(
        monitor="val_accuracy", 
        min_delta=0.00, 
        patience=3, 
        verbose=False, 
        mode="max",
    )
    training_data_monitor_callback = TrainingDataMonitor(
        log_every_n_steps=25,
    )
    batch_gradient_verification_callback = BatchGradientVerificationCallback()

    wandb_logger = WandbLogger(
        name=exp_name,
        project='comp579-jellybean',
    )

    dm = JellybeanDataModule(
        dataset_file=dataset_file,
        batch_size=hparams.batch_size,
    )
    dm.setup()

    trainer = Trainer.from_argparse_args(
        args, 
        callbacks=[
            checkpoint_callback, 
            device_stats_monitor_callback, 
            early_stop_callback, 
            progress_bar_callback, 
            print_table_metrics_callback, 
            training_data_monitor_callback, 
            batch_gradient_verification_callback,
        ],
        logger=WandbLogger
    )
    model = model_class(num_stack=hparams.num_stack)
    trainer.fit(
        model, 
        dm.train_dataloader(), 
        dm.val_dataloader(),
    )
    trainer.save_checkpoint()
    del model

    model = model_class()
    trainer.validate(
        model,
        dataloaders=[dm.val_dataloader()],
        ckpt=hparams.default_root_dir,
    )
    trainer.test(
        model,
        dataloaders=[dm.test_dataloader()],
        ckpt=hparams.default_root_dir,
    )
    model = MyLightningModule.load_from_checkpoint(
        checkpoint_file=hparams.default_root_dir,
    )
    trainer.predict(
        model,
        dataloaders=[dm.val_dataloader()], 
        return_predictions=True, 
        ckpt_path=hparams.default_root_dir,
    )



if __name__ == "__main__":
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument("--num_stack", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--mode", type=str, default="simple")
    args = parser.parse_args()

    main(args)