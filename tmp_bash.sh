# MIRNN (d=128)
python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mirnn model.hidden_size=128 train.batch_size=128

# MIRNN (d=256)
python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mirnn model.hidden_size=256 train.batch_size=128

# MIRNN (d=512)
python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mirnn model.hidden_size=512 train.batch_size=128

# MIRNN (d=1024)
python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mirnn model.hidden_size=1024 train.batch_size=128

# MIRNN (d=2048)
python train.py data.path=$SLURM_TMPDIR/data/processed/ptb model.name=mirnn model.hidden_size=2048 train.batch_size=128