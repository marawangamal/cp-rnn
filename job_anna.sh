#!/bin/bash
#SBATCH --cpus-per-task=2                                # Ask for 2 CPUs
#SBATCH --gres=gpu:4                                     # Ask for 1 GPU
#SBATCH --mem=10G                                        # Ask for 10 GB of RAM
#SBATCH --time=3:00:00                                   # The job will run for 3 hours
#SBATCH -o /network/scratch/<u>/<username>/slurm-%j.out  # Write the log on scratch

# 1. Load the required modules
module load python/3.8

# 2. Load your environment
source /home/mila/m/marawan.gamal/.venv/cv/bin/activate

# 3. Make visible
export CUDA_VISIBLE_DEVICES=0,1

# 3. Copy your dataset on the compute node
cp -r data $SLURM_TMPDIR/data

# 4. Launch your job, tell it to save the model in $SLURM_TMPDIR and look for the dataset into $SLURM_TMPDIR


## ===============================================================================================================
## Experiment A | HP Tuning MIRNN (d=2048) | Character Level
## ===============================================================================================================
#
## Vary BS
#python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=mirnn model.hidden_size=2048 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=mirnn model.hidden_size=2048 train.batch_size=256
#python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=mirnn model.hidden_size=2048 train.batch_size=512
#python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=mirnn model.hidden_size=2048 train.batch_size=1024
#
## Vary Dropout
#python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=mirnn model.hidden_size=2048 model.dropout=0
#python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=mirnn model.hidden_size=2048 model.dropout=0.1
#python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=mirnn model.hidden_size=2048 model.dropout=0.5

# ===============================================================================================================
# Experiment B | Compare Different Models | Character Level
# ===============================================================================================================

# Every line is a variation of dimension. (Diff lines for different ranks

## 2RNN
#python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=2rnn model.hidden_size=128 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=2rnn model.hidden_size=256 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=2rnn model.hidden_size=512 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=2rnn model.hidden_size=1024 train.batch_size=128
#
## MIRNN (d=128)
#python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=mirnn model.hidden_size=128 train.batch_size=128
#
## MIRNN (d=256)
#python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=mirnn model.hidden_size=256 train.batch_size=128
#
## MIRNN (d=512)
#python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=mirnn model.hidden_size=512 train.batch_size=128
#
## MIRNN (d=1024)
#python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=mirnn model.hidden_size=1024 train.batch_size=128
#
## MIRNN (d=2048)
#python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=mirnn model.hidden_size=2048 train.batch_size=128

## MRNN (d=128)
#python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=mrnn model.hidden_size=128 model.rank=32 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=mrnn model.hidden_size=128 model.rank=64 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=mrnn model.hidden_size=128 model.rank=128 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=mrnn model.hidden_size=128 model.rank=256 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=mrnn model.hidden_size=128 model.rank=512 train.batch_size=128
#
## MRNN (d=256)
#python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=mrnn model.hidden_size=256 model.rank=32 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=mrnn model.hidden_size=256 model.rank=64 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=mrnn model.hidden_size=256 model.rank=128 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=mrnn model.hidden_size=256 model.rank=256 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=mrnn model.hidden_size=256 model.rank=512 train.batch_size=128
#
## MRNN (d=512)
#python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=mrnn model.hidden_size=512 model.rank=32 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=mrnn model.hidden_size=512 model.rank=64 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=mrnn model.hidden_size=512 model.rank=128 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=mrnn model.hidden_size=512 model.rank=256 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=mrnn model.hidden_size=512 model.rank=512 train.batch_size=128
#
## MRNN (d=1024)
#python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=mrnn model.hidden_size=1024 model.rank=32 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=mrnn model.hidden_size=1024 model.rank=64 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=mrnn model.hidden_size=1024 model.rank=128 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=mrnn model.hidden_size=1024 model.rank=256 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=mrnn model.hidden_size=1024 model.rank=512 train.batch_size=128
#
## MRNN (d=2048)
#python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=mrnn model.rank=32 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=mrnn model.rank=64 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=mrnn model.rank=128 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=mrnn model.rank=256 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=mrnn model.rank=512 train.batch_size=128

## CPRNN (d=128)
python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=cprnn model.hidden_size=128 model.rank=32 train.batch_size=128
python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=cprnn model.hidden_size=128 model.rank=64 train.batch_size=128
python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=cprnn model.hidden_size=128 model.rank=128 train.batch_size=128
python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=cprnn model.hidden_size=128 model.rank=256 train.batch_size=128
python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=cprnn model.hidden_size=128 model.rank=512 train.batch_size=128
python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=cprnn model.hidden_size=128 model.rank=1024 train.batch_size=128
python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=cprnn model.hidden_size=128 model.rank=2048 train.batch_size=128
python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=cprnn model.hidden_size=128 model.rank=4096 train.batch_size=128

# CPRNN (d=256)
python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=cprnn model.hidden_size=256 model.rank=32 train.batch_size=128
python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=cprnn model.hidden_size=256 model.rank=64 train.batch_size=128
python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=cprnn model.hidden_size=256 model.rank=128 train.batch_size=128
python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=cprnn model.hidden_size=256 model.rank=256 train.batch_size=128
python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=cprnn model.hidden_size=256 model.rank=512 train.batch_size=128
python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=cprnn model.hidden_size=256 model.rank=1024 train.batch_size=128
python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=cprnn model.hidden_size=256 model.rank=2048 train.batch_size=128
python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=cprnn model.hidden_size=256 model.rank=4096 train.batch_size=128

# CPRNN (d=512)
python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=cprnn model.hidden_size=512 model.rank=32 train.batch_size=128
python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=cprnn model.hidden_size=512 model.rank=64 train.batch_size=128
python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=cprnn model.hidden_size=512 model.rank=128 train.batch_size=128
python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=cprnn model.hidden_size=512 model.rank=256 train.batch_size=128
python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=cprnn model.hidden_size=512 model.rank=512 train.batch_size=128
python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=cprnn model.hidden_size=512 model.rank=1024 train.batch_size=128
python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=cprnn model.hidden_size=512 model.rank=2048 train.batch_size=128
python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=cprnn model.hidden_size=512 model.rank=4096 train.batch_size=128

# CPRNN (d=1024)
python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=cprnn model.hidden_size=1024 model.rank=32 train.batch_size=128
python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=cprnn model.hidden_size=1024 model.rank=64 train.batch_size=128
python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=cprnn model.hidden_size=1024 model.rank=128 train.batch_size=128
python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=cprnn model.hidden_size=1024 model.rank=256 train.batch_size=128
python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=cprnn model.hidden_size=1024 model.rank=512 train.batch_size=128
python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=cprnn model.hidden_size=1024 model.rank=1024 train.batch_size=128
python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=cprnn model.hidden_size=1024 model.rank=2048 train.batch_size=128
python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=cprnn model.hidden_size=1024 model.rank=4096 train.batch_size=128

# CPRNN (d=2048)
python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=cprnn model.rank=32 train.batch_size=128
python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=cprnn model.rank=64 train.batch_size=128
python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=cprnn model.rank=128 train.batch_size=128
python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=cprnn model.rank=256 train.batch_size=128
python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=cprnn model.rank=512 train.batch_size=128
python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=cprnn model.rank=1024 train.batch_size=128
python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=cprnn model.rank=2048 train.batch_size=128
python train.py data.path=$SLURM_TMPDIR/data/processed/anna model.name=cprnn model.rank=4096 train.batch_size=128
