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
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 3. Copy your dataset on the compute node
cp -r data $SLURM_TMPDIR/data

# 4. Launch your job, tell it to save the model in $SLURM_TMPDIR
#    and look for the dataset into $SLURM_TMPDIR
python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.rank=32
#python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=cprnn model.rank=64
