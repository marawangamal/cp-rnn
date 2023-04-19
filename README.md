# CP-RNN

This repository is the official implementation of [Paper Title](https://arxiv.org/abs/2030.12345). 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Usage

**Build Data**
To generate toy datasets:
```cmd
python cprnn/features/build_features.py -d toy
```

To build PTB:
```cmd
python cprnn/features/build_features.py -d ptb
```

**Train**

To train a model on toy datasets:
```train
python train.py -d data/processed/toy-rnn-i8-h8-v4-r4  
```

To train on PTB
```train
python train.py -d data/processed/ptb  
```

**Evaluate**

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
