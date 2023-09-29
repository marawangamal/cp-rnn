#!/bin/bash
#SBATCH --cpus-per-task=2                                # Ask for 2 CPUs
#SBATCH --gres=gpu:1                                     # Ask for 1 GPU
#SBATCH --mem=10G                                        # Ask for 10 GB of RAM
#SBATCH --time=5:00:00                                   # The job will run for 3 hours


# 1. Load the required modules
module load python/3.8

# 2. Load your environment
source /home/mila/l/lizairem/virtualenvs/cprnn/bin/activate

# 3. Make visible
export CUDA_VISIBLE_DEVICES=0,1

# 3. Copy your dataset on the compute node
#$SCRATCH/data $SLURM_TMPDIR/data
cp -r data $SLURM_TMPDIR/data

# 4. Launch your job, tell it to save the model in $SLURM_TMPDIR and look for the dataset into $SLURM_TMPDIR


## ===============================================================================================================
## Experiment A | HP Tuning MIRNN (d=2048) | Character Level
## ===============================================================================================================
#
## Vary BS
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mirnn model.hidden_size=2048 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mirnn model.hidden_size=2048 train.batch_size=256
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mirnn model.hidden_size=2048 train.batch_size=512
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mirnn model.hidden_size=2048 train.batch_size=1024
#
## Vary Dropout
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mirnn model.hidden_size=2048 model.dropout=0
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mirnn model.hidden_size=2048 model.dropout=0.1
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mirnn model.hidden_size=2048 model.dropout=0.5

# ===============================================================================================================
# Experiment B | Compare Different Models | Character Level
# ===============================================================================================================

# Every line is a variation of dimension. (Diff lines for different ranks

## 2RNN
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=2rnn model.hidden_size=4 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=2rnn model.hidden_size=8 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=2rnn model.hidden_size=16 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=2rnn model.hidden_size=32 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=2rnn model.hidden_size=64 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=2rnn model.hidden_size=128 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=2rnn model.hidden_size=256 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=2rnn model.hidden_size=512 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=2rnn model.hidden_size=1024 train.batch_size=128

## MIRNN (d={11,34,97,245,561,1203,2494})
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mirnn model.hidden_size=11 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mirnn model.hidden_size=34 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mirnn model.hidden_size=97 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mirnn model.hidden_size=245 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mirnn model.hidden_size=561 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mirnn model.hidden_size=1203 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mirnn model.hidden_size=2494 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mirnn model.hidden_size=5079 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mirnn model.hidden_size=10249 train.batch_size=128
#
## MIRNNNew (d={11,34,97,245,561,1203,2494})
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mirnnnew model.hidden_size=11 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mirnnnew model.hidden_size=34 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mirnnnew model.hidden_size=97 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mirnnnew model.hidden_size=245 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mirnnnew model.hidden_size=561 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mirnnnew model.hidden_size=1203 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mirnnnew model.hidden_size=2494 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mirnnnew model.hidden_size=5079 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mirnnnew model.hidden_size=10249 train.batch_size=128
#

python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=birnn model.hidden_size=256 train.batch_size=128
python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=rnn model.hidden_size=256 train.batch_size=128

#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mirnnnew model.hidden_size=1203 train.batch_size=64
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mirnnnew model.hidden_size=1203 train.batch_size=256 train.lr=0.001 train.factor=0.02 train.threshold=0.02 train.patience=3 train.seq_len=150
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=2rnn model.hidden_size=128 train.batch_size=64 train.lr=0.001 train.factor=0.02 train.threshold=0.02 train.patience=3 train.seq_len=150
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=2rnn model.hidden_size=128 train.batch_size=128 train.lr=0.001 train.factor=0.02 train.threshold=0.02 train.patience=3 train.seq_len=150
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=2rnn model.hidden_size=128 train.batch_size=128 train.lr=0.001 train.factor=0.02 train.threshold=0.02 train.patience=3 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=2rnn model.hidden_size=128 train.batch_size=128 train.lr=0.002 train.factor=0.02 train.threshold=0.02 train.patience=3 train.seq_len=150

#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mirnnnew model.hidden_size=1203 train.batch_size=128 train.lr=0.001 train.factor=0.02 train.threshold=0.02 train.patience=3 train.seq_len=100
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mirnnnew model.hidden_size=1203 train.batch_size=128 train.lr=0.001 train.factor=0.02 train.threshold=0.02 train.patience=3 train.seq_len=150


## CPRNN (number params = 29376, h_2rnn=16)
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=16 model.rank=193 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=32 model.rank=132 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=64 model.rank=53 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=90 model.rank=10 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=45 model.rank=95 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=60 model.rank=61 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=75 model.rank=34 train.batch_size=128

## CPRNN (number params = 430848, h_2rnn=64)
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=32 model.rank=2565 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=128 model.rank=1088 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=512 model.rank=57 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=256 model.rank=511 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=300 model.rank=399 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=330 model.rank=335 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=400 model.rank=210 train.batch_size=128

## CPRNN (number params = 6736896, h_2rnn=256)
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=128 model.rank=18752 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=512 model.rank=5662 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=2048 model.rank=506 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=1024 model.rank=2550 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=1300 model.rank=1770 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=1450 model.rank=1445 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=1600 model.rank=1166 train.batch_size=128

## MIRNNNew (exp fixed hidden size)
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mirnnnew model.hidden_size=64 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mirnnnew model.hidden_size=256 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mirnnnew model.hidden_size=1024 train.batch_size=128

## CPRNN (exp fixed hidden size, H=64, rank={64, 128, 256, 512, 1024, 2048})
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=64 model.rank=16 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=64 model.rank=32 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=64 model.rank=64 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=64 model.rank=128 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=64 model.rank=256 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=64 model.rank=512 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=64 model.rank=1024 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=64 model.rank=2048 train.batch_size=128

## CPRNN (exp fixed hidden size, H=256, rank={128, 256, 512, 1024, 2048})
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=256 model.rank=64 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=256 model.rank=128 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=256 model.rank=256 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=256 model.rank=512 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=256 model.rank=1024 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=256 model.rank=2048 train.batch_size=128

## CPRNN (exp fixed hidden size, H=1024, rank={128, 256, 512, 1024, 2048})
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=1024 model.rank=128 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=1024 model.rank=256 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=1024 model.rank=512 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=1024 model.rank=1024 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=1024 model.rank=2048 train.batch_size=128

## CPRNN (exp fixed ranks{64, 256, 1024}, hidden={32, 2048})
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=32 model.rank=64 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=2048 model.rank=64 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=32 model.rank=256 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=2048 model.rank=256 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=32 model.rank=1024 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=2048 model.rank=1024 train.batch_size=128

## MIRNN (d=32)
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mirnn model.hidden_size=32 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mirnn model.hidden_size=128 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mirnn model.hidden_size=256 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mirnn model.hidden_size=512 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mirnn model.hidden_size=1024 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mirnn model.hidden_size=2048 train.batch_size=128


## MRNN (d=128)
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mrnn model.hidden_size=128 model.rank=32 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mrnn model.hidden_size=128 model.rank=64 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mrnn model.hidden_size=128 model.rank=128 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mrnn model.hidden_size=128 model.rank=256 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mrnn model.hidden_size=128 model.rank=512 train.batch_size=128
#
## MRNN (d=256)
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mrnn model.hidden_size=256 model.rank=32 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mrnn model.hidden_size=256 model.rank=64 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mrnn model.hidden_size=256 model.rank=128 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mrnn model.hidden_size=256 model.rank=256 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mrnn model.hidden_size=256 model.rank=512 train.batch_size=128
#
## MRNN (d=512)
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mrnn model.hidden_size=512 model.rank=32 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mrnn model.hidden_size=512 model.rank=64 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mrnn model.hidden_size=512 model.rank=128 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mrnn model.hidden_size=512 model.rank=256 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mrnn model.hidden_size=512 model.rank=512 train.batch_size=128
#
## MRNN (d=1024)
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mrnn model.hidden_size=1024 model.rank=32 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mrnn model.hidden_size=1024 model.rank=64 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mrnn model.hidden_size=1024 model.rank=128 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mrnn model.hidden_size=1024 model.rank=256 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mrnn model.hidden_size=1024 model.rank=512 train.batch_size=128
#
## MRNN (d=2048)
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mrnn model.rank=32 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mrnn model.rank=64 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mrnn model.rank=128 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mrnn model.rank=256 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mrnn model.rank=512 train.batch_size=128

# CPRNN (d=32)
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=32 model.rank=32 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=32 model.rank=64 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=32 model.rank=128 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=32 model.rank=256 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=32 model.rank=512 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=32 model.rank=1024 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=32 model.rank=2048 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=32 model.rank=4096 train.batch_size=128


# CPRNN (d=128)
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=128 model.rank=32 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=128 model.rank=64 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=128 model.rank=128 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=128 model.rank=256 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=128 model.rank=512 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=128 model.rank=1024 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=128 model.rank=2048 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=128 model.rank=4096 train.batch_size=128

# CPRNN (d=256)
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=256 model.rank=32 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=256 model.rank=64 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=256 model.rank=128 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=256 model.rank=256 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=256 model.rank=512 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=256 model.rank=1024 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=256 model.rank=2048 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=256 model.rank=4096 train.batch_size=128

# CPRNN (d=512)
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=512 model.rank=32 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=512 model.rank=64 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=512 model.rank=128 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=512 model.rank=256 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=512 model.rank=512 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=512 model.rank=1024 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=512 model.rank=2048 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=512 model.rank=4096 train.batch_size=128

# CPRNN (d=1024)
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=1024 model.rank=32 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=1024 model.rank=64 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=1024 model.rank=128 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=1024 model.rank=256 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=1024 model.rank=512 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=1024 model.rank=1024 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=1024 model.rank=2048 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=1024 model.rank=4096 train.batch_size=128

# CPRNN (d=2048)
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.rank=32 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.rank=64 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.rank=128 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.rank=256 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.rank=512 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.rank=1024 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.rank=2048 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.rank=4096 train.batch_size=128
