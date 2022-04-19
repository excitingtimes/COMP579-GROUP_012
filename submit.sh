#!/bin/bash
#SBATCH --output=/home/mila/s/sonnery.hugo/scratch/logs/jellyean/%A-output.out
#SBATCH --error=/home/mila/s/sonnery.hugo/scratch/logs/jellyean/%A-error.out
#SBATCH --time=1-00:00:00
#SBATCH --job-name=remote
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=48G
#SBATCH --cpus-per-gpu=4
#SBATCH --signal=SIGUSR1@90 
#SBATCH --mail-user=<sonnery.hugo@mila.quebec>
#SBATCH --partition=long
#SBATCH --array=0-7

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

export NUM_STACK=(
    1
    2
    4
    8
    12
    16
    24
    32
)

python train.py \
    --default_root_dir="/home/mila/s/sonnery.hugo/scratch/checkpoints/jellyean/" \
    --num_stack=${NUM_STACK[${SLURM_ARRAY_TASK_ID}]} \
    --batch_size=64 \
    --max_epochs=100 \
    --gpus=1 \
    --mode="simple"