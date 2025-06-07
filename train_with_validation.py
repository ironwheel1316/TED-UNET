import argparse
import os
from collections import OrderedDict
from glob import glob
import cv2


import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
import albumentations as A
from albumentations import Compose, OneOf
from albumentations import RandomRotate90
from albumentations import ShiftScaleRotate
from albumentations import GaussianBlur
from albumentations import ElasticTransform
from albumentations.augmentations.transforms import (
    HueSaturationValue, RandomBrightness, 
    RandomContrast, Normalize
)
from albumentations import Flip
from albumentations import Resize
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm
import matplotlib.pyplot as plt

import archs
import losses
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter, str2bool
# python train_with_validation.py --dataset dsb2018_96 --arch UNet
# python train_with_validation.py --dataset dsb2018_96 --arch NestedUNet
# python train_with_validation.py --dataset dsb2018_96 --arch UNetPlusMinus
# python train_with_validation.py --dataset dsb2018_96 --arch UNetPlusMinus2
# python train_with_validation.py --dataset dsb2018_96 --arch WNet
# python train_with_validation.py --dataset dsb2018_96 --arch WNetPlusPlus
# python train_with_validation.py --dataset dsb2018_96 --arch UNetDeep
# python train_with_validation.py --dataset dsb2018_96 --arch Conv3UNet

# python train_with_validation.py --dataset kvasir_seg_96 --arch UNet3 --input_w 96 --input_h 96 --batch_size 18
# python train_with_validation.py --dataset kvasir_seg_96 --arch NestedUNet
# python train_with_validation.py --dataset kvasir_seg_96 --arch UNetPlusMinus
# python train_with_validation.py --dataset kvasir_seg_96 --arch UNetPlusMinus2
# python train_with_validation.py --dataset kvasir_seg_96 --arch WNet
# python train_with_validation.py --dataset kvasir_seg_96 --arch UNetR1
# python train_with_validation.py --dataset kvasir_seg_96 --arch UNetR12
# python train_with_validation.py --dataset kvasir_seg_96 --arch WNetPlusPlus
# python train_with_validation.py --dataset kvasir_seg_96 --arch UNetDeep
# python train_with_validation.py --dataset kvasir_seg_96 --arch Conv3UNet
# python train_with_validation.py --dataset kvasir_seg_96 --arch ConvUNet123
# python train_with_validation.py --dataset kvasir_seg_96 --arch ConvUNet321
# python train_with_validation.py --dataset kvasir_seg_96 --arch ConvUNet31122
# python train_with_validation.py --dataset kvasir_seg_96 --arch ConvUNet32222
# python train_with_validation.py --dataset kvasir_seg_96 --arch ConvUNet32211
# python train_with_validation.py --dataset kvasir_seg_96 --arch ConvUNet3333
# python train_with_validation.py --dataset kvasir_seg_96 --arch ConvUNet444
# python train_with_validation.py --dataset kvasir_seg_96 --arch ConvUNet55
# python train_with_validation.py --dataset kvasir_seg_96 --arch ConvUNet6
# python train_with_validation.py --dataset kvasir_seg_96 --arch R1UNet444UP --input_w 96 --input_h 96 --batch_size 18



# python train_with_validation.py --dataset cityscape_96 --arch UNet 
# python train_with_validation.py --dataset cityscape_96 --arch NestedUNet
# python train_with_validation.py --dataset cityscape_96 --arch UNetPlusMinus
# python train_with_validation.py --dataset cityscape_96 --arch UNetPlusMinus2
# python train_with_validation.py --dataset cityscape_96 --arch WNet
# python train_with_validation.py --dataset cityscape_96 --arch WNetPlusPlus
# python train_with_validation.py --dataset cityscape_96 --arch UNetDeep
# python train_with_validation.py --dataset cityscape_96 --arch Conv3UNet
# python train_with_validation.py --dataset cityscape_96 --arch ConvUNet123
# python train_with_validation.py --dataset cityscape_96 --arch ConvUNet321
# python train_with_validation.py --dataset cityscape_96 --arch ConvUNet31122
# python train_with_validation.py --dataset cityscape_96 --arch ConvUNet32222
# python train_with_validation.py --dataset cityscape_96 --arch ConvUNet32211
# python train_with_validation.py --dataset cityscape_96 --arch ConvUNet3333
# python train_with_validation.py --dataset cityscape_96 --arch ConvUNet444
# python train_with_validation.py --dataset cityscape_96 --arch ConvUNet55
# python train_with_validation.py --dataset cityscape_96 --arch ConvUNet6

# python train_with_validation.py --dataset cityscape_224 --arch UNet --input_w 224 --input_h 224
# python train_with_validation.py --dataset cityscape_224--arch NestedUNet --input_w 224 --input_h 224
# python train_with_validation.py --dataset cityscape_224 --arch UNetPlusMinus --input_w 224 --input_h 224
# python train_with_validation.py --dataset cityscape_224 --arch UNetPlusMinus2 --input_w 224 --input_h 224
# python train_with_validation.py --dataset cityscape_224 --arch WNet --input_w 224 --input_h 224
# python train_with_validation.py --dataset cityscape_224 --arch WNetPlusPlus --input_w 224 --input_h 224
# python train_with_validation.py --dataset cityscape_224 --arch UNetDeep --input_w 224 --input_h 224
# python train_with_validation.py --dataset cityscape_224 --arch Conv3UNet --input_w 224 --input_h 224  
# python train_with_validation.py --dataset cityscape_224 --arch ConvUNet123 --input_w 224 --input_h 224
# python train_with_validation.py --dataset cityscape_224 --arch ConvUNet321 --input_w 224 --input_h 224
# python train_with_validation.py --dataset cityscape_224 --arch ConvUNet31122 --input_w 224 --input_h 224
# python train_with_validation.py --dataset cityscape_224 --arch ConvUNet32222 --input_w 224 --input_h 224
# python train_with_validation.py --dataset cityscape_224 --arch ConvUNet32221 --input_w 224 --input_h 224
# python train_with_validation.py --dataset cityscape_224 --arch ConvUNet3333 --input_w 224 --input_h 224
# python train_with_validation.py --dataset cityscape_224 --arch ConvUNet444 --input_w 224 --input_h 224
# python train_with_validation.py --dataset cityscape_224 --arch ConvUNet55 --input_w 224 --input_h 224
# python train_with_validation.py --dataset cityscape_224 --arch ConvUNet6 --input_w 224 --input_h 224


# python train_with_validation.py --dataset kvasir_seg_288 --arch UNet
# python train_with_validation.py --dataset kvasir_seg_288 --arch NestedUNet
# python train_with_validation.py --dataset kvasir_seg_288 --arch UNetPlusMinus
# python train_with_validation.py --dataset kvasir_seg_288 --arch UNetPlusMinus2
# python train_with_validation.py --dataset kvasir_seg_288 --arch WNet
# python train_with_validation.py --dataset kvasir_seg_288 --arch WNetPlusPlus
# python train_with_validation.py --dataset kvasir_seg_288 --arch UNetDeep
# python train_with_validation.py --dataset kvasir_seg_288 --arch Conv3UNet
# python train_with_validation.py --dataset kvasir_seg_288 --arch ConvUNet123
# python train_with_validation.py --dataset kvasir_seg_288 --arch ConvUNet321
# python train_with_validation.py --dataset kvasir_seg_288 --arch ConvUNet31122
# python train_with_validation.py --dataset kvasir_seg_288 --arch ConvUNet32222
# python train_with_validation.py --dataset kvasir_seg_96 --arch ConvUNet11223
# python train_with_validation.py --dataset kvasir_seg_288 --arch ConvUNet3333
# python train_with_validation.py --dataset kvasir_seg_288 --arch ConvUNet444
# python train_with_validation.py --dataset kvasir_seg_288 --arch ConvUNet55
# python train_with_validation.py --dataset kvasir_seg_288 --arch ConvUNet6
# python train_with_validation.py --dataset kvasir_seg_288 --arch ResUNet

# python train_with_validation.py --dataset clinicdb_96 --arch ConvUNet3333

ARCH_NAMES = archs.__all__
LOSS_NAMES = losses.__all__

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=150, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=16, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    
    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='NestedUNet',
                        choices=ARCH_NAMES,
                        help='model architecture: ' +
                        ' | '.join(ARCH_NAMES) +
                        ' (default: NestedUNet)')
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=96, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=96, type=int,
                        help='image height')
    
    # loss
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=LOSS_NAMES,
                        help='loss: ' +
                        ' | '.join(LOSS_NAMES) +
                        ' (default: BCEDiceLoss)')
    
    # dataset
    parser.add_argument('--dataset', default='dsb2018_96',
                        help='dataset name')
    parser.add_argument('--img_ext', default='.png',
                        help='image file extension')
    parser.add_argument('--mask_ext', default='.png',
                        help='mask file extension')

    # optimizer
    parser.add_argument('--optimizer', default='SGD',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                        ' | '.join(['Adam', 'SGD']) +
                        ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2/3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')
    
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--validation_interval', default=10, type=int,
                        help='number of epochs between validations')

    config = parser.parse_args()

    return config


def train(config, train_loader, model, criterion, optimizer, epoch):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    model.train()
    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        input = input.cuda()
        target = target.cuda()

        if config['deep_supervision']:
            outputs = model(input)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)
            iou = iou_score(outputs[-1], target)
        else:
            output = model(input)
            loss = criterion(output, target)
            iou = iou_score(output, target)

        optimizer.zero_grad()
        loss.backward()
        # After loss.backward()
    
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])


def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.cuda()
            target = target.cuda()

            if config['deep_supervision']:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou = iou_score(outputs[-1], target)
            else:
                output = model(input)
                loss = criterion(output, target)
                iou = iou_score(output, target)

            Ascend = True
            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])


def plot_results(log, model_name):
    epochs = log['epoch']
    train_loss = log['loss']
    train_iou = log['iou']
    val_loss = log['val_loss']
    val_iou = log['val_iou']
    val_epochs = log['val_epoch']

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(val_epochs, val_loss, label='Validation Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_iou, label='Train IoU')
    plt.plot(val_epochs, val_iou, label='Validation IoU', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.title('Training and Validation IoU')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'models/{model_name}/training_plot.png')
    plt.close()


def main():
    config = vars(parse_args())
    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_wDS_%s' % (config['dataset'], config['arch'], timestamp)
        else:
            config['name'] = '%s_%s_woDS_%s' % (config['dataset'], config['arch'], timestamp)
    os.makedirs('models/%s' % config['name'], exist_ok=True)

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    with open('models/%s/config.yml' % config['name'], 'w') as f:
        yaml.dump(config, f)

    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    elif config['loss'] == 'FocalLoss':
        criterion = losses.FocalLoss(alpha=0.5, gamma=0, ignore_index=config['num_classes'] - 1).cuda()
    else:
        criterion = losses.__dict__[config['loss']]().cuda()

    cudnn.benchmark = True

    print("=> creating model %s" % config['arch'])
    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                        #    config['deep_supervision']
    )

    model = model.cuda()

    params = filter(lambda p: p.requires_grad, model.parameters())
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                   verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')], gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    # 以8:1:1的比例划分训练集、验证集和测试集
    train_img_ids, temp_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)
    val_img_ids, test_img_ids = train_test_split(temp_img_ids, test_size=0.5, random_state=41)
    
    # 保存测试集ID到文件，以便后续测试使用
    with open(f'models/{config["name"]}/test_ids.txt', 'w') as f:
        for img_id in test_img_ids:
            f.write(f"{img_id}\n")
    print(f"Test set IDs saved to models/{config['name']}/test_ids.txt")

    train_transform = Compose([
        RandomRotate90(),
        Flip(),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.15, rotate_limit=45, p=0.5, border_mode=cv2.BORDER_CONSTANT),
        # ElasticTransform(p=0.3, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, border_mode=cv2.BORDER_CONSTANT),
        OneOf([
            HueSaturationValue(),
            RandomBrightness(),
            RandomContrast(),
        ], p=1),
        # OneOf([
        #     GaussianBlur(blur_limit=(3, 7), p=0.5),
        #     A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        # ], p=0.4),
        Resize(config['input_h'], config['input_w']),
        Normalize(),
    ])

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        Normalize(),
    ])

    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=train_transform)
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)
    
    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('val_loss', []),
        ('val_iou', []),
        ('val_epoch', [])
    ])

    best_iou = 0
    trigger = 0
    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))

        train_log = train(config, train_loader, model, criterion, optimizer,epoch)
        
        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()

        print('loss %.4f - iou %.4f' % (train_log['loss'], train_log['iou']))

        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])

        # 确保val_loss, val_iou和val_epoch的长度与epoch一致
        if (epoch + 1) % config['validation_interval'] == 0:
            val_log = validate(config, val_loader, model, criterion)
            if config['scheduler'] == 'ReduceLROnPlateau':
                scheduler.step(val_log['loss'])

            print('val_loss %.4f - val_iou %.4f' % (val_log['loss'], val_log['iou']))

            log['val_loss'].append(val_log['loss'])
            log['val_iou'].append(val_log['iou'])
            log['val_epoch'].append(epoch)
        else:
            # 如果不是验证轮次，添加None或上一次的验证结果
            if log['val_loss']:
                log['val_loss'].append(log['val_loss'][-1])
                log['val_iou'].append(log['val_iou'][-1])
                log['val_epoch'].append(log['val_epoch'][-1])
            else:
                log['val_loss'].append(None)
                log['val_iou'].append(None)
                log['val_epoch'].append(None)

        pd.DataFrame(log).to_csv('models/%s/log.csv' % config['name'], index=False)
        print("=> saved log")

        trigger += 1

        if (epoch + 1) % config['validation_interval'] == 0 and log.get('val_iou', [0])[-1] is not None and log.get('val_iou', [0])[-1] > best_iou:
            torch.save(model.state_dict(), 'models/%s/model.pth' % config['name'])
            best_iou = log['val_iou'][-1]
            print("=> saved best model")
            trigger = 0

        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()

    plot_results(log, config['name'])
    print("=> saved training plot")


if __name__ == '__main__':
    main()
