# TED-UNET
Up to now, many variants have emerged based on UNET to solve the problem of image cutting. However, we found that UNET is still the one with the best generalization. Therefore, we returned to the most basic UNET architecture again, hoping to make changes on this basis. Thus, we discovered TE-UNET

This repository contains code for a image segmentation model based on [UNet++: A Nested U-Net Architecture for Medical Image Segmentation](https://arxiv.org/abs/1807.10165) implemented in PyTorch.



This repository contains architecture ['UNet', 'NestedUNet', 'UNetPlusMinus','UNetPlusMinus2','WNet','WNetPlusPlus', 'UNetDeep', 'Conv3UNet','ConvUNet11223', 'ConvUNet32211', 'ConvUNet31122', 'ConvUNet32222', 'ConvUNet32221', 'ConvUNet3333', 'ConvUNet444', 'ConvUNet55','ConvUNet6',"ResUNet","UNetR1","UNetR12", "R1UNet444","R1UNet444UP","R1UNet444UP333","NOR1UNet444","UNet3"]


<img width="1100" alt="3f3f3c4c23612207dfac8b7876c26c5" src="https://github.com/user-attachments/assets/de46c16a-e2d0-48f6-9a0b-daf4569406d5" />

[**NEW**] Add support for multi-class segmentation dataset.

[**NEW**] Add support for PyTorch 1.x.


## Requirements
- PyTorch 1.x or 0.41

## Installation
1. Create an anaconda environment.
```sh
conda create -n=<env_name> python=3.6 anaconda
conda activate <env_name>
```
2. Install PyTorch.
```sh
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```
3. Install pip packages.
```sh
pip install -r requirements.txt
```

## Training on [Kvasir-SEG](https://www.kaggle.com/datasets/debeshjha1/kvasirseg) dataset
1. Download dataset from [here](https://www.kaggle.com/datasets/debeshjha1/kvasirseg) to inputs/ and unzip. The file structure is the following:
```
inputs
└── Kvasir-SEG
        ├── images
        │     └── cju0qk...
        └── masks
              └── cju0qk...           
                     ...
```
2. Preprocess.
```sh
python preprocess_kvasir_seg.py
```
3. Train the model.
```sh
python train.py --dataset kvasir_96 --arch NestedUNet
```
4. Evaluate.
```sh
python val.py --name kvasir_96_NestedUNet_woDS_(timestamp)
```
### (Optional) Using LovaszHingeLoss
1. Clone LovaszSoftmax from [bermanmaxim/LovaszSoftmax](https://github.com/bermanmaxim/LovaszSoftmax).
```
git clone https://github.com/bermanmaxim/LovaszSoftmax.git
```
2. Train the model with LovaszHingeLoss.
```
python train.py --dataset dsb2018_96 --arch NestedUNet --loss LovaszHingeLoss
```

## Training on original dataset
Make sure to put the files as the following structure (e.g. the number of classes is 2):
```
inputs
└── <dataset name>
    ├── images
    |   ├── 0a7e06.jpg
    │   ├── 0aab0a.jpg
    │   ├── 0b1761.jpg
    │   ├── ...
    |
    └── masks
        ├── 0
        |   ├── 0a7e06.png
        |   ├── 0aab0a.png
        |   ├── 0b1761.png
        |   ├── ...
        |
        └── 1
            ├── 0a7e06.png
            ├── 0aab0a.png
            ├── 0b1761.png
            ├── ...
```

1. Train the model.
```
python train.py --dataset <dataset name> --arch NestedUNet --img_ext .jpg --mask_ext .png
```
2. Evaluate.
```
python val.py --name <dataset name>_NestedUNet_woDS
```

## Results
### Kvasir-SEG (96x96)

Here is the results on Kvasir-SEG dataset (96x96).

| Model                           |   IoU   |  Loss   |
|:------------------------------- |:-------:|:-------:|
| U-Net                           |  0.700  |  0.322  |
| Nested U-Net                    |  0.699  |**0.320**|
| TED-UNET（R1UNet444)            |**0.768**|  0.264  |


![232f417d296a5fe6af03603aa339469](https://github.com/user-attachments/assets/95e3b299-2be5-49a0-b7ba-aceaafa657f4)



