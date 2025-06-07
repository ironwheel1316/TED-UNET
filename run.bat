


@echo off
echo Activating conda environment...
call C:\Users\Administrator\anaconda3\Scripts\activate.bat C:\Users\Administrator\anaconda3\envs\unet++
echo Conda environment activated.

@REM echo Starting training with UNet architecture for kvasir_seg_96 dataset...
@REM python train_with_validation.py --dataset clinicdb_288 --arch UNet 
@REM echo All training processes completed.

echo Starting training with UNet architecture for kvasir_seg_96 dataset...
python train_with_validation.py --dataset clinicdb_288 --arch R1UNet444 
echo Training with UNet completed.

echo Starting training with UNet architecture for kvasir_seg_96 dataset...
python train_with_validation.py --dataset clinicdb_288 --arch UNetR1 
echo Training with UNet completed.



echo Starting training with UNet architecture for kvasir_seg_96 dataset...
python train_with_validation.py --dataset clinicdb_288 --arch UNetR12 

echo Starting training with UNet architecture for kvasir_seg_96 dataset...


echo Starting training with UNet architecture for kvasir_seg_96 dataset...
python train_with_validation.py --dataset clinicdb_288 --arch ConvUNet444 


@REM echo Starting training with UNet architecture for kvasir_seg_96 dataset...
@REM python train_with_validation.py --dataset clinicdb_96 --arch UNetDeep 
@REM echo Training with UNet completed.


echo Starting training with UNet architecture for kvasir_seg_96 dataset...
python train_with_validation.py --dataset clinicdb_288 --arch UNetPlusMinus 

echo Starting training with UNet architecture for kvasir_seg_96 dataset...
python train_with_validation.py --dataset clinicdb_288 --arch UNetPlusMinus2 
echo Training with UNet completed.


echo Starting training with UNet architecture for kvasir_seg_96 dataset...
python train_with_validation.py --dataset clinicdb_288 --arch ConvUNet32211 
echo Training with UNet completed.

echo Starting training with UNet architecture for kvasir_seg_96 dataset...
python train_with_validation.py --dataset clinicdb_288 --arch ConvUNet11223 

echo Starting training with UNet architecture for kvasir_seg_96 dataset...
python train_with_validation.py --dataset clinicdb_288 --arch Conv3UNet 
echo Training with UNet completed.


echo Starting training with UNet architecture for kvasir_seg_96 dataset...
python train_with_validation.py --dataset clinicdb_288 --arch NestedUNet --batch_size 8
echo All training processes completed.

echo Starting training with UNet architecture for kvasir_seg_96 dataset...
python train_with_validation.py --dataset clinicdb_288 --arch UNet3 
echo All training processes completed.

pause





@REM python train_with_validation.py --dataset clinicdb_96 --arch UNet
@REM python train_with_validation.py --dataset clinicdb_96 --arch NestedUNet
@REM python train_with_validation.py --dataset clinicdb_96 --arch UNetPlusMinus
@REM python train_with_validation.py --dataset clinicdb_96 --arch UNetPlusMinus2
@REM python train_with_validation.py --dataset clinicdb_96 --arch WNet
@REM python train_with_validation.py --dataset clinicdb_96 --arch WNetPlusPlus
@REM python train_with_validation.py --dataset clinicdb_96 --arch UNetDeep
@REM python train_with_validation.py --dataset clinicdb_96 --arch Conv3UNet
@REM python train_with_validation.py --dataset clinicdb_96 --arch ConvUNet123
@REM python train_with_validation.py --dataset clinicdb_96 --arch ConvUNet33221
@REM python train_with_validation.py --dataset clinicdb_96 --arch ConvUNet31122
@REM python train_with_validation.py --dataset clinicdb_96 --arch ConvUNet32222
@REM python train_with_validation.py --dataset clinicdb_96 --arch ConvUNet32221
@REM python train_with_validation.py --dataset clinicdb_96 --arch ConvUNet3333
@REM python train_with_validation.py --dataset clinicdb_96 --arch ConvUNet444
@REM python train_with_validation.py --dataset clinicdb_96 --arch ConvUNet55
@REM python train_with_validation.py --dataset clinicdb_96 --arch ConvUNet6