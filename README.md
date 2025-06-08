### TED-UNet

UNet has been widely adopted for biomedical image segmentation tasks due to their strong performance and flexibility. After 10 years of its publication, UNet still displays its superiority over other models and even some of its optimized variants. In this work, we deliberately choose to build upon the original UNet architecture and propose TED-UNet, a lightweight yet effective optimization based on UNet. Experimental results on the *Kvasir-SEG* dataset demonstrate improved performance with significantly fewer parameters for our model, making room for further developments and improvements.

This repository contains code for a image segmentation model based on [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) [UNet++: A Nested U-Net Architecture for Medical Image Segmentation](https://arxiv.org/abs/1807.10165) implemented by PyTorch on Windows.



The model contains architecture ['UNet', 'NestedUNet', 'UNetPlusMinus','UNetPlusMinus2','WNet','WNetPlusPlus', 'UNetDeep', 'Conv3UNet','ConvUNet11223', 'ConvUNet32211', 'ConvUNet31122', 'ConvUNet32222', 'ConvUNet32221', 'ConvUNet3333', 'ConvUNet444', 'ConvUNet55','ConvUNet6',"ResUNet","UNetR1","UNetR12", "R1UNet444"(TED-UNet),"R1UNet444UP","R1UNet444UP333","NOR1UNet444","UNet3"]

![image](https://github.com/user-attachments/assets/b69ccb8a-2732-4624-9d06-910bd6a7eab9)


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
3. Train and evluate the model.
```sh
python train_with_validation.py --dataset kvasir_96 --arch NestedUNet
```
4. Test the model.
```sh
python test.py --name kvasir_96_NestedUNet_woDS_(timestamp)
```
5. Test all the model in kvasir_96 dataset.
```sh
python batch_test_and_plot.py
(Then, input dataset name , e.g. kvasir_96)
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

1. Train and evluate the model.
```
python train_with_validation.py --dataset <dataset name> --arch NestedUNet --dataset <Yours>
```
2. Test the model.
```
python val.py --name <dataset name>_NestedUNet_woDS
```
## Tool Instructions

| File                            |                                                       function                                                         | 
|:------------------------------- |:----------------------------------------------------------------------------------------------------------------------:|
| archs                           | All models we used.                                                                                                    | 
| input                           | The location where our dataset is stored.                                                                              | 
| models                          | I provided pth that could display all the indicators.                                                                  | 
| train_with_validation           | The training is verified every ten rounds to achieve a better observation of whether overfitting has occurred.         |
| calculate_flops                 | Analyze the parameters, storage volume and computational load of this model.                                           | 
| batch_test_and_plot             | Test all the models under a certain dataset and generate a comparison chart.                                           | 
| compare_logs                    | At the same time, for all the models in all the datasets, generate the IOU and Loss values for training and validation.| 
| preprocess_dsb2018              | Preprocess the corresponding dataset.Download dataset from [here](https://www.kaggle.com/c/data-science-bowl-2018/data)| 
| preprocess_kvasir_seg           | Preprocess the corresponding dataset.Download dataset from [here](https://www.kaggle.com/datasets/debeshjha1/kvasirseg)| 
| preprocess_ClinicDB             | Preprocess the corresponding dataset..Download dataset from [here](https://www.kaggle.com/datasets/orvile/cvc-clinicdb)| 
| test_output                     | Stored the results of the test set.                                                                                    | 
| run.bat                         | A script for completing multiple trainings at once.                                                                    | 




## Results
### Kvasir-SEG (96x96)

Here is the results on Kvasir-SEG dataset (96x96).

| Model                           |   IoU   |  Loss   |
|:------------------------------- |:-------:|:-------:|
| U-Net                           |  0.700  |  0.322  |
| Nested U-Net                    |  0.699  |**0.320**|
| TED-UNET（R1UNet444)            |**0.768**|  0.264  |


<img width="1274" alt="截屏2025-06-07 21 31 12" src="https://github.com/user-attachments/assets/031cefc8-ddb0-4dc2-9946-1c27057f1fb3" />


![fedd390a47a74040cf36fbcae1f952e](https://github.com/user-attachments/assets/0bf4bada-dfce-4c5d-bae4-af66e50deee1)

![image](https://github.com/user-attachments/assets/8b42d455-7d40-4187-a7e3-5ecd752c35cd)



