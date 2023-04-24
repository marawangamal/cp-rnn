# CP-RNN

This repository is the official implementation of [Paper Title](https://arxiv.org/abs/2030.12345). 

## Installation

Install package and dependencies:

```setup
python setup.py install
pip install -r requirements.txt
```

## Build data

To generate Penn Tree Bank dataset, Anna dataset or a Toy dataset run:

```cmd
python cprnn/features/build_features.py -d [toy/ptb/anna]
```
 

## Train

Enter training configs in `configs.yaml` then run the command below. Results will be saved in `runs` folder.

```train
python train.py
```

## Visualize
```train
tensorboard --logdir=runs
```

## Sample Outputs

CP-RNN Rank 128 @ Epoch 50

`The took said a come in investment it were all admines in the coming thrual and spock industrial stock o
`

CP-RNN Rank 16 @ Epoch 50

`Then they in monced this and the as anessue otton thestences a n't as big sayernment agricens is costsha
`

