#!/bin/bash
#SBATCH --output=/home/mila/s/sonnery.hugo/scratch/logs/jellyean/%A-output.out
#SBATCH --error=/home/mila/s/sonnery.hugo/scratch/logs/jellyean/%A-error.out
#SBATCH --time=2-00:00:00
#SBATCH --job-name=remote
#SBATCH --gres=gpu:4
#SBATCH --mem-per-gpu=48G
#SBATCH --cpus-per-gpu=4
#SBATCH --signal=SIGUSR1@90 
#SBATCH --mail-user=<sonnery.hugo@mila.quebec>
#SBATCH --partition=long
#SBATCH --array=0-0

# A few useful environment variables
export WANDB_API_KEY="48e1f847d659d327cf9148cfc864f4200506c728" # Please do not share !
export DATA_PATH="/home/mila/s/sonnery.hugo/scratch/datasets/jellyean/"
export CHECKPOINT_DIR="/home/mila/s/sonnery.hugo/scratch/checkpoints/jellyean/"
export LOGS_DIR="/home/mila/s/sonnery.hugo/scratch/logs/jellyean/"
export WANDB_SAVE_DIR="/home/mila/s/sonnery.hugo/scratch/outputs/jellyean/"
export TMPDIR="/home/mila/s/sonnery.hugo/scratch/outputs/jellyean/"
export SID=${SLURM_ARRAY_TASK_ID}
export HYDRA_FULL_ERROR=1
export PYTHONDONTWRITEBYTECODE=1 # Prevent Python from creating __pycache__

# Load a pre-created Conda environment
module load anaconda/3
module load cuda/11.2
conda activate env-rl

# Create the environment :
conda env create environment.yml -n env-rl
conda activate env-rl

cd jelly-bean-world/api/python
python setup.py install

conda install -c conda-forge gym==0.22.0 
conda install pytorch-lightning
conda install pytorch-lightning-bolts

conda env config vars set LD_LIBRARY_PATH=/usr/local/pkgs/cuda/latest/lib64:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia-460:/usr/lib/nvidia

export NUM_STACK=(
    1
    2
    4
    8
    16
)
${NUM_STACK[${SLURM_ARRAY_TASK_ID}]}

python train.py \
    --default_root_dir="/home/mila/s/sonnery.hugo/scratch/checkpoints/jellyean/" \
    -num_stack=1 \
    -batch_size=64 \
    -mode="simple"