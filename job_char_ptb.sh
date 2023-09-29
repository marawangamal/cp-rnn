#!/bin/bash
#SBATCH --cpus-per-task=2                                # Ask for 2 CPUs
#SBATCH --gres=gpu:1                                     # Ask for 1 GPU
#SBATCH --mem=10G                                        # Ask for 10 GB of RAM
#SBATCH --time=4:00:00                                   # The job will run for 3 hours


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
## Experiment A | HP Tuning to reproduce paper MIRNN | Character Level | Penn Tree bank
## ===============================================================================================================

#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=rnn model.hidden_size=2048 model.dropout=0 train.batch_size=32 train.lr=0.001 train.factor=0.5 train.threshold=0.02 train.patience=3 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=rnn model.hidden_size=2048 model.dropout=0 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.02 train.patience=3 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=rnn model.hidden_size=2048 model.dropout=0.2 train.batch_size=32 train.lr=0.001 train.factor=0.5 train.threshold=0.02 train.patience=3 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=rnn model.hidden_size=2048 model.dropout=0.2 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.02 train.patience=3 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=rnn model.hidden_size=2048 model.dropout=0.4 train.batch_size=32 train.lr=0.001 train.factor=0.5 train.threshold=0.02 train.patience=3 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=rnn model.hidden_size=2048 model.dropout=0.4 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.02 train.patience=3 train.seq_len=50

#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mirnn model.hidden_size=2048 model.dropout=0 train.batch_size=32 train.lr=0.001 train.factor=0.5 train.threshold=0.02 train.patience=3 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mirnn model.hidden_size=2048 model.dropout=0 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.02 train.patience=3 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mirnn model.hidden_size=2048 model.dropout=0.2 train.batch_size=32 train.lr=0.001 train.factor=0.5 train.threshold=0.02 train.patience=3 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mirnn model.hidden_size=2048 model.dropout=0.2 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.02 train.patience=3 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mirnn model.hidden_size=2048 model.dropout=0.4 train.batch_size=32 train.lr=0.001 train.factor=0.5 train.threshold=0.02 train.patience=3 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mirnn model.hidden_size=2048 model.dropout=0.4 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.02 train.patience=3 train.seq_len=50

#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=2rnn model.hidden_size=256 model.dropout=0 train.batch_size=32 train.lr=0.001 train.factor=0.5 train.threshold=0.02 train.patience=3 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=2rnn model.hidden_size=256 model.dropout=0 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.02 train.patience=3 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=2rnn model.hidden_size=256 model.dropout=0.2 train.batch_size=32 train.lr=0.001 train.factor=0.5 train.threshold=0.02 train.patience=3 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=2rnn model.hidden_size=256 model.dropout=0.2 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.02 train.patience=3 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=2rnn model.hidden_size=256 model.dropout=0.4 train.batch_size=32 train.lr=0.001 train.factor=0.5 train.threshold=0.02 train.patience=3 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=2rnn model.hidden_size=256 model.dropout=0.4 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.02 train.patience=3 train.seq_len=50

## ===============================================================================================================
## Experiment B | HP Fine Tuning to reproduce paper MIRNN | Character Level | Penn Tree bank
## ===============================================================================================================

#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=2rnn model.hidden_size=256 model.dropout=0 train.batch_size=32 train.lr=0.0001 train.factor=0.5 train.threshold=0.02 train.patience=1 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=2rnn model.hidden_size=256 model.dropout=0 train.batch_size=128 train.lr=0.0001 train.factor=0.5 train.threshold=0.02 train.patience=1 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=2rnn model.hidden_size=256 model.dropout=0 train.batch_size=32 train.lr=0.001 train.factor=0.1 train.threshold=0.04 train.patience=1 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=2rnn model.hidden_size=256 model.dropout=0 train.batch_size=128 train.lr=0.001 train.factor=0.1 train.threshold=0.04 train.patience=1 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=2rnn model.hidden_size=256 model.dropout=0 train.batch_size=32 train.lr=0.001 train.factor=0.5 train.threshold=0.04 train.patience=1 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=2rnn model.hidden_size=256 model.dropout=0 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.04 train.patience=1 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=2rnn model.hidden_size=256 model.dropout=0 train.batch_size=32 train.lr=0.001 train.factor=0.5 train.threshold=0.04 train.patience=2 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=2rnn model.hidden_size=256 model.dropout=0 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.04 train.patience=2 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=2rnn model.hidden_size=256 model.dropout=0 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.03 train.patience=2 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=2rnn model.hidden_size=256 model.dropout=0 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.01 train.patience=2 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=2rnn model.hidden_size=256 model.dropout=0 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.02 train.patience=1 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=2rnn model.hidden_size=256 model.dropout=0 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.02 train.patience=2 train.seq_len=50



#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mirnn model.hidden_size=2048 model.dropout=0 train.batch_size=128 train.lr=0.001 train.factor=0.05 train.threshold=0.02 train.patience=1 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mirnn model.hidden_size=2048 model.dropout=0 train.batch_size=128 train.lr=0.001 train.factor=0.1 train.threshold=0.04 train.patience=2 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mirnn model.hidden_size=2048 model.dropout=0 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.05 train.patience=2 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mirnn model.hidden_size=2048 model.dropout=0 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.04 train.patience=1 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mirnn model.hidden_size=2048 model.dropout=0 train.batch_size=128 train.lr=0.001 train.factor=0.1 train.threshold=0.04 train.patience=1 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mirnn model.hidden_size=2048 model.dropout=0 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.01 train.patience=2 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mirnn model.hidden_size=2048 model.dropout=0 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.03 train.patience=2 train.seq_len=50

#
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=rnn model.hidden_size=2048 model.dropout=0 train.batch_size=128 train.lr=0.001 train.factor=0.05 train.threshold=0.02 train.patience=1 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=rnn model.hidden_size=2048 model.dropout=0 train.batch_size=128 train.lr=0.001 train.factor=0.1 train.threshold=0.04 train.patience=2 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=rnn model.hidden_size=2048 model.dropout=0 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.05 train.patience=2 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=rnn model.hidden_size=2048 model.dropout=0 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.04 train.patience=1 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=rnn model.hidden_size=2048 model.dropout=0 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.04 train.patience=2 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=rnn model.hidden_size=2048 model.dropout=0 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.03 train.patience=2 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=rnn model.hidden_size=2048 model.dropout=0 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.02 train.patience=2 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=rnn model.hidden_size=2048 model.dropout=0 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.01 train.patience=2 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=rnn model.hidden_size=2048 model.dropout=0 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.02 train.patience=1 train.seq_len=50


#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=2048 model.rank=506 train.batch_size=128 train.lr=0.001 train.factor=0.05 train.threshold=0.02 train.patience=1 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=2048 model.rank=506 train.batch_size=128 train.lr=0.001 train.factor=0.1 train.threshold=0.04 train.patience=2 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=2048 model.rank=506 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.05 train.patience=2 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=2048 model.rank=506 train.batch_size=32 train.lr=0.001 train.factor=0.05 train.threshold=0.02 train.patience=1 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=2048 model.rank=506 train.batch_size=32 train.lr=0.001 train.factor=0.1 train.threshold=0.04 train.patience=2 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=2048 model.rank=506 train.batch_size=32 train.lr=0.001 train.factor=0.5 train.threshold=0.05 train.patience=2 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=2rnn model.hidden_size=256 model.dropout=0 train.batch_size=32 train.lr=0.0001 train.factor=0.5 train.threshold=0.02 train.patience=1 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=2rnn model.hidden_size=256 model.dropout=0 train.batch_size=128 train.lr=0.0001 train.factor=0.5 train.threshold=0.02 train.patience=1 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=2048 model.rank=506 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.04 train.patience=1 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=2048 model.rank=506 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.01 train.patience=2 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=2048 model.rank=506 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.03 train.patience=2 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=2048 model.rank=506 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.02 train.patience=2 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=2048 model.rank=506 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.02 train.patience=1 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=2048 model.rank=506 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.04 train.patience=4 train.seq_len=50

## ===============================================================================================================
## Experiment C | Verify HP Tuning for different hs | Character Level
## ===============================================================================================================
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=rnn model.hidden_size=32 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.02 train.patience=2 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=rnn model.hidden_size=32 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.04 train.patience=2 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=rnn model.hidden_size=32 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.02 train.patience=1 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=rnn model.hidden_size=32 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.04 train.patience=1 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=rnn model.hidden_size=128 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.02 train.patience=2 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=rnn model.hidden_size=128 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.04 train.patience=2 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=rnn model.hidden_size=128 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.02 train.patience=1 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=rnn model.hidden_size=128 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.04 train.patience=1 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=rnn model.hidden_size=512 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.02 train.patience=2 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=rnn model.hidden_size=512 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.04 train.patience=2 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=rnn model.hidden_size=512 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.02 train.patience=1 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=rnn model.hidden_size=512 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.04 train.patience=1 train.seq_len=50

#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=2rnn model.hidden_size=16 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.02 train.patience=2 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=2rnn model.hidden_size=16 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.04 train.patience=2 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=2rnn model.hidden_size=16 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.02 train.patience=1 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=2rnn model.hidden_size=16 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.04 train.patience=1 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=2rnn model.hidden_size=64 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.02 train.patience=2 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=2rnn model.hidden_size=64 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.04 train.patience=2 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=2rnn model.hidden_size=64 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.02 train.patience=1 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=2rnn model.hidden_size=64 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.04 train.patience=1 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=2rnn model.hidden_size=4 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.02 train.patience=2 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=2rnn model.hidden_size=4 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.04 train.patience=2 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=2rnn model.hidden_size=4 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.02 train.patience=1 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=2rnn model.hidden_size=4 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.04 train.patience=1 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=2rnn model.hidden_size=128 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.02 train.patience=2 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=2rnn model.hidden_size=128 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.04 train.patience=2 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=2rnn model.hidden_size=128 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.02 train.patience=1 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=2rnn model.hidden_size=128 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.04 train.patience=1 train.seq_len=50
python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=2rnn model.hidden_size=1024 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.02 train.patience=2 train.seq_len=50
python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=2rnn model.hidden_size=1024 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.04 train.patience=2 train.seq_len=50
python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=2rnn model.hidden_size=1024 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.02 train.patience=1 train.seq_len=50
python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=2rnn model.hidden_size=1024 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.04 train.patience=1 train.seq_len=50

#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=32 model.rank=33 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.01 train.patience=2 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=32 model.rank=33 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.02 train.patience=2 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=32 model.rank=33 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.04 train.patience=2 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=32 model.rank=33 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.01 train.patience=1 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=32 model.rank=33 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.02 train.patience=1 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=32 model.rank=33 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.04 train.patience=1 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=128 model.rank=180 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.02 train.patience=2 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=128 model.rank=180 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.04 train.patience=2 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=128 model.rank=180 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.02 train.patience=1 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=128 model.rank=180 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.04 train.patience=1 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=512 model.rank=520 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.02 train.patience=2 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=512 model.rank=520 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.04 train.patience=2 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=512 model.rank=520 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.02 train.patience=1 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=512 model.rank=520 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.04 train.patience=1 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=128 model.rank=180 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.01 train.patience=2 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=512 model.rank=520 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.01 train.patience=2 train.seq_len=50

#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mirnn model.hidden_size=32 model.dropout=0 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.05 train.patience=2 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mirnn model.hidden_size=128 model.dropout=0 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.05 train.patience=2 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mirnn model.hidden_size=512 model.dropout=0 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.05 train.patience=2 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mirnn model.hidden_size=32 model.dropout=0 train.batch_size=128 train.lr=0.001 train.factor=0.1 train.threshold=0.04 train.patience=1 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mirnn model.hidden_size=128 model.dropout=0 train.batch_size=128 train.lr=0.001 train.factor=0.1 train.threshold=0.04 train.patience=1 train.seq_len=50
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mirnn model.hidden_size=512 model.dropout=0 train.batch_size=128 train.lr=0.001 train.factor=0.1 train.threshold=0.04 train.patience=1 train.seq_len=50

# ===============================================================================================================
# Experiment D | CPRNN : fixed number of params | Character Level PTB
# ===============================================================================================================
## CPRNN (number params = 29461, h_2rnn=16)
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=4 model.rank=261 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=16 model.rank=194 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=32 model.rank=132 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=45 model.rank=95 train.batch_size=128
##python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=60 model.rank=61 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=64 model.rank=54 train.batch_size=128
##python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=75 model.rank=34 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=90 model.rank=11 train.batch_size=128

## CPRNN (number params = 430885, h_2rnn=64)
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=4 model.rank=3945 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=16 model.rank=3213 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=32 model.rank=2565 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=64 model.rank=1807 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=101 model.rank=1320 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=128 model.rank=1088 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=256 model.rank=511 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=512 model.rank=57 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=300 model.rank=399 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=330 model.rank=335 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=400 model.rank=210 train.batch_size=128

## CPRNN (number params = 6736741, h_2rnn=256)
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=64 model.rank=29343 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=90 model.rank=23880 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=101 model.rank=22132 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=128 model.rank=18751 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=256 model.rank=10798 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=512 model.rank=5663 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=1024 model.rank=2550 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=1300 model.rank=1771 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=1450 model.rank=1446 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=1600 model.rank=1167 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.hidden_size=2048 model.rank=507 train.batch_size=128



# ===============================================================================================================
# Experiment E | Compare Different Models | Character Level PTB
# ===============================================================================================================

# Every line is a variation of dimension.

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

## MIRNN (hs={11,34,97,246,561,1204,2495, 5079, 10249})
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mirnn model.hidden_size=11 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mirnn model.hidden_size=34 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mirnn model.hidden_size=97 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mirnn model.hidden_size=246 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mirnn model.hidden_size=561 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mirnn model.hidden_size=1204 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mirnn model.hidden_size=2495 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mirnn model.hidden_size=5079 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mirnn model.hidden_size=10249 train.batch_size=128

## RNN (hs={11,34,97,247,563,1205,2496, 5080, 10251})
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=rnn model.hidden_size=11 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=rnn model.hidden_size=34 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=rnn model.hidden_size=98 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=rnn model.hidden_size=247 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=rnn model.hidden_size=563 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=rnn model.hidden_size=1205 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=rnn model.hidden_size=2496 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=rnn model.hidden_size=5080 train.batch_size=128
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=rnn model.hidden_size=10251 train.batch_size=128
#
# ===============================================================================================================
# Experiment F | Compare Different Models | Character Level PTB
# ===============================================================================================================

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
