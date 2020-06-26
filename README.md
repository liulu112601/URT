# A Universal Representation Transformer Layer for Few-Shot Image Classification


## Dependencies
This code requires the following:
* Python 3.6 or greater
* PyTorch 1.0 or greater
* TensorFlow 1.14 or greater


## Data Preparation 
1. Meta-Dataset:

    Follow the the "User instructions" in the [Meta-Dataset repository](https://github.com/google-research/meta-dataset#user-instructions) for "Installation" and "Downloading and converting datasets".

2. Additional Test Datasets:

    If you want to test on additional datasets, i.e.,  MNIST, CIFAR10, CIFAR100, follow the installation instructions in the [CNAPs repository](https://github.com/cambridge-mlg/cnaps) to get these datasets.

## Getting the Feature Extractors

URT can be built on top of backbones pretrained in any ways. 

The easiest way is to download SUR's pre-trained models and use them to obtain a universal set of features directly. If that is what you want, execute the following command in the root directory of this project:```wget http://thoth.inrialpes.fr/research/SUR/all_weights.zip && unzip all_weights.zip && rm all_weights.zip```
It will donwnload all the weights and place them in the `./weights` directory.
Or pretrain the backbone by yourself on the training sets of Meta-Dataset and put the model weights under the directory of `./weights`. 

## Train and evaluate URT


### Dumping features (for efficient training and evaluation)
We found that the bottleneck of training URT is extracting features from CNN. Since we freeze the CNN when training the URT, we found dumping the extracted feature episodes can significantly speed up the training procedure from days to ~2 hours. The easiest way is to download all the extracted features from [HERE](https://drive.google.com/drive/folders/1Z3gsa4TSSiH2wTZj1Jp5bD7UEKPOVzx5?usp=sharing) and put it in the ${cache_dir}.
Or you can extract by your own via ```bash ./scripts/pre-extract-feature.sh resnet18 ${cache_dir}```

### Train and evaluate
run command from the dir of this repo: ```bash ./fast-scripts/urt-avg-head.sh ${log_dir} ${num_head} ${penalty_coef} ${cache_dir}```, where the ${num_head}=2 and ${penalty_coef}=0.1 in our paper.
