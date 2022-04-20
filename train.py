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
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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
    def __init__(self, dataset_file, num_stack=1, n_samples=1_000_000):
        self.dataset_file = dataset_file
        if self.dataset_file is not None:
            print(self.dataset_file)
            data = torch.load(f=self.dataset_file)
            self.scent, self.vision, self.feature, self.episode = data["scent"], data["vision"], data["feature"], data["episode"]
        else:
            self.num_stack = num_stack
            self.n_samples = n_samples
            self.delta_stack = 5
            self.max_samples_per_episode = 1_000

            self.env = gym.make("JBW-COMP579-obj-v1", render=False)
            self.stacked_env = FrameStack(self.env, num_stack=self.num_stack, lz4_compress=False)
            _ = self.stacked_env.reset()

            self._dim_scent_ = [self.num_stack, 3]
            self._dim_vision_ = [self.num_stack, 16, 16, 3]
            self._dim_feature_ = [self.num_stack, 900]

    def __len__(self):
        if self.dataset_file is not None:
            return len(self.scent)
        else:
            return self.n_samples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.dataset_file is not None:
            sample = {
                'scent': self.scent[idx], 
                'vision': rearrange(self.vision[idx], 'num_stack h w c -> (num_stack c) h w'),
                'feature': self.feature[idx],
                'episode': self.episode[idx],
            }
        else:
            n_gathered_samples = 0
            delta = 0
            i = 0

            dim_scent = [len(idx)] + self._dim_scent_
            dim_vision = [len(idx)] + self._dim_vision_
            dim_feature = [len(idx)] + self._dim_feature_
            dim_episode = [len(idx), 1]

            gathered_scent = torch.zeros(dim_scent)
            gathered_vision = torch.zeros(dim_vision)
            gathered_feature = torch.zeros(dim_feature)
            gathered_episode = torch.zeros(dim_episode)

            while not done:
                svfm, reward, done, info  = self.stacked_env.step(self.stacked_env.action_space.sample())
                scent, vision, feature, moved = svfm
                delta += 1
                # Every {delta_stack} frames, we stack the {num_stack} frames together and add them to our dataset
                if (delta % self.delta_stack) == 0:
                    i += 1
                    gathered_scent[i] = torch.Tensor(scent[:])
                    gathered_vision[i, :, :15, :15, :] = torch.Tensor(vision[:])
                    gathered_feature[i] = torch.Tensor(feature[:])
                    gathered_episode[i] = delta
                    # For every {max_samples_per_episode} samples collected in our dataloader, we reset the environment to sample a new trajectory
                    if (i % self.max_samples_per_episode) == 0:
                        _ = self.stacked_env.reset()

                if (i == self.n_samples - 1):
                    break

            sample = {
                'scent': gathered_scent, 
                'vision': rearrange(gathered_vision, 'num_stack h w c -> (num_stack c) h w'),
                'feature': gathered_feature[idx],
                'episode': gathered_episode[idx]
            }

        return sample['vision'], sample['episode']

class JellybeanDataModule(pl.LightningDataModule):
    def __init__(self, dataset_file="/home/mila/s/sonnery.hugo/scratch/datasets/jellyean/dataset-jellybean-num_stack=1.pt", batch_size=64, num_stack=1, n_samples=1_000_000):
        super().__init__()
        self.dataset_file = dataset_file
        self.batch_size = batch_size
        self.num_stack = num_stack
        self.n_samples = n_samples

    def setup(self, stage=None):
        self.jellybean_full = JellybeanDataset(
            self.dataset_file,
            num_stack=self.num_stack,
            n_samples=self.n_samples,
        )
        if self.dataset_file is not None:
            self.jellybean_train, self.jellybean_val, self.jellybean_test = random_split(self.jellybean_full, [7_000, 2_000, 1_000])
        else:
            n_train = int(0.7 * self.n_samples)
            n_val = int(0.2 * self.n_samples)
            n_test = self.n_samples - n_train - n_val
            self.jellybean_train, self.jellybean_val, self.jellybean_test = random_split(self.jellybean_full, [n_train, n_val, n_test])

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



class Encoder(nn.Module):
    def __init__(self, num_input_channels: int, base_channel_size: int, latent_dim: int, act_fn: object = nn.GELU):
        """
        Args:
           num_input_channels : Number of input channels of the image. For CIFAR, this parameter is 3
           base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
           latent_dim : Dimensionality of latent representation z
           act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        c_hid = base_channel_size
        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2),  # 32x32 => 16x16
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 16x16 => 8x8
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 8x8 => 4x4
            act_fn(),
            nn.Flatten(),  # Image grid to single feature vector
            nn.Linear(2 * 4 * c_hid, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, num_input_channels: int, base_channel_size: int, latent_dim: int, act_fn: object = nn.GELU):
        """
        Args:
           num_input_channels : Number of channels of the image to reconstruct. For CIFAR, this parameter is 3
           base_channel_size : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
           latent_dim : Dimensionality of latent representation z
           act_fn : Activation function used throughout the decoder network
        """
        super().__init__()
        c_hid = base_channel_size
        self.linear = nn.Sequential(nn.Linear(latent_dim, 2 * 4 * c_hid), act_fn())
        self.net = nn.Sequential(
            nn.ConvTranspose2d(
                2 * c_hid, 2 * c_hid, kernel_size=3, output_padding=1, padding=1, stride=2
            ),  # 4x4 => 8x8
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(2 * c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),  # 8x8 => 16x16
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(
                c_hid, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2
            ),  # 16x16 => 32x32
            nn.Tanh(),  # The input images is scaled between -1 and 1, hence the output has to be bounded as well
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 2, 2)
        x = self.net(x)
        return x



class Autoencoder(pl.LightningModule):
    def __init__(
        self,
        base_channel_size: int,
        latent_dim: int,
        encoder_class: object = Encoder,
        decoder_class: object = Decoder,
        num_input_channels: int = 3,
        width: int = 32,
        height: int = 32,
    ):
        super().__init__()
        # Saving hyperparameters of autoencoder
        self.save_hyperparameters()
        # Creating encoder and decoder
        self.encoder = encoder_class(num_input_channels, base_channel_size, latent_dim)
        self.decoder = decoder_class(num_input_channels, base_channel_size, latent_dim)
        # Example input array needed for visualizing the graph of the network
        self.example_input_array = torch.zeros(2, num_input_channels, width, height)

    def forward(self, x):
        """The forward function takes in an image and returns the reconstructed image."""
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def _get_reconstruction_loss(self, batch):
        """Given a batch of images, this function returns the reconstruction loss (MSE in our case)"""
        x, _ = batch  # We do not need the labels
        x_hat = self.forward(x)
        loss = F.mse_loss(x, x_hat, reduction="none")
        loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=20, min_lr=5e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("test_loss", loss)



def main(hparams):
    save_file = f"dataset-jellybean-num_stack={hparams.num_stack}.pt"
    save_directory = "/home/mila/s/sonnery.hugo/scratch/datasets/jellyean/"
    if hparams.live:
        dataset_file = None
    else:
        dataset_file = save_directory + save_file
    exp_name = f"autoencoder-num_stack={str(hparams.num_stack)}-mode={hparams.mode}"

    hparams.num_stack = int(hparams.num_stack)

    """
    model_class = lambda num_stack: AE(
        input_height=15, 
        enc_type='resnet18', 
        first_conv=False, 
        maxpool1=False, 
        enc_out_dim=512, 
        latent_dim=256, 
        lr=0.0001,
    )
    """
    model_class = lambda num_stack: Autoencoder(
        base_channel_size=64,
        latent_dim=128,
        encoder_class=Encoder,
        decoder_class=Decoder,
        num_input_channels=3*hparams.num_stack,
        width=15,
        height=15,
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
        monitor="val_loss", 
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
        log_model='all',
        save_dir='/home/mila/s/sonnery.hugo/scratch/outputs/jellyean/wandb',
    )
    model = model_class(num_stack=hparams.num_stack)
    wandb_logger.watch(model, log="all")

    dm = JellybeanDataModule(
        dataset_file=dataset_file,
        batch_size=hparams.batch_size,
        num_stack=hparams.num_stack,
        n_samples=hparams.n_samples,
    )
    dm.setup()

    trainer = Trainer.from_argparse_args(
        args, 
        callbacks=[
            checkpoint_callback, 
            device_stats_monitor_callback, 
            # early_stop_callback, 
            progress_bar_callback, 
            print_table_metrics_callback, 
            training_data_monitor_callback, 
            batch_gradient_verification_callback,
        ],
        logger=wandb_logger,
    )
    trainer.fit(
        model, 
        dm.train_dataloader(), 
        dm.val_dataloader(),
    )
    del model

    """
    model = model_class(num_stack=hparams.num_stack).load_from_checkpoint(
        checkpoint_path=hparams.default_root_dir,
    )
    trainer.validate(
        model,
        dataloaders=[dm.val_dataloader()],
        ckpt_path=hparams.default_root_dir,
    )
    trainer.test(
        model,
        dataloaders=[dm.test_dataloader()],
        ckpt_path=hparams.default_root_dir,
    )
    trainer.predict(
        model,
        dataloaders=[dm.val_dataloader()], 
        return_predictions=True, 
        ckpt_path=hparams.default_root_dir,
    )
    """



if __name__ == "__main__":
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument("--num_stack", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--mode", type=str, default="simple")
    parser.add_argument("--live", type=bool, default=False)
    parser.add_argument("--n_samples", type=int, default=1_000_000)
    args = parser.parse_args()

    main(args)