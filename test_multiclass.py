import argparse
import os
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
import numpy as np
import albumentations as A
from collections import OrderedDict
from albumentations import Resize, Normalize, Compose
from glob import glob
from sklearn.model_selection import train_test_split
import cv2

import archs
import losses
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter, str2bool

# python test_multiclass.py --model_dir models/<model_name> --dataset <dataset_name>
# Example: python test_multiclass.py --model_dir cityscape_96_UNet_woDS_20250529_042132 --dataset cityscape_96 --arch UNet --num_classes 20
# python test_multiclass.py --model_dir cityscape_224_UNet_woDS_20250530_130642 --dataset cityscape_96 --arch UNet --num_classes 20
ARCH_NAMES = archs.__all__
LOSS_NAMES = losses.__all__

# 类别颜色映射（可以根据需要调整颜色）
COLOR_MAP = [
(128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153),
    (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152),
    (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
    (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32), (0, 0, 0) 
]

def parse_args():
    parser = argparse.ArgumentParser(description='Test model on test set for multiclass segmentation.')
    parser.add_argument('--model_dir', type=str, required=True, 
                        help='Directory containing the trained model.')
    parser.add_argument('--dataset', type=str, required=True, 
                        help='Dataset name to test on.')
    parser.add_argument('--test_ids_file', type=str, default=None,
                        help='Path to a file containing test image IDs. If not provided, will look for test_ids.txt in model_dir.')
    parser.add_argument('--img_ext', type=str, default='.png', 
                        help='Image file extension.')
    parser.add_argument('--mask_ext', type=str, default='.png', 
                        help='Mask file extension.')
    parser.add_argument('--input_w', type=int, default=96,
                        help='Image width.')
    parser.add_argument('--input_h', type=int, default=96,
                        help='Image height.')
    parser.add_argument('--num_classes', type=int, default=20, 
                        help='Number of classes.')
    parser.add_argument('--batch_size', type=int, default=16, 
                        help='Batch size for testing.')
    parser.add_argument('--num_workers', type=int, default=4, 
                        help='Number of workers for data loading.')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='NestedUNet',
                        choices=ARCH_NAMES,
                        help='Model architecture: ' +
                        ' | '.join(ARCH_NAMES) +
                        ' (default: NestedUNet)')
    parser.add_argument('--input_channels', type=int, default=3,
                        help='Input channels.')
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--loss', default='FocalLoss',
                        choices=LOSS_NAMES,
                        help='Loss function: ' +
                        ' | '.join(LOSS_NAMES) +
                        ' (default: FocalLoss)')

    config = parser.parse_args()
    return config

def load_test_ids(model_dir, test_ids_file=None):
    if test_ids_file:
        test_ids_path = test_ids_file
    else:
        test_ids_path = os.path.join('models', model_dir, 'test_ids.txt')
    
    # 确保路径使用正确的分隔符
    test_ids_path = os.path.abspath(test_ids_path)
    print(f"Looking for test IDs file at: {test_ids_path}")
    
    if os.path.exists(test_ids_path):
        with open(test_ids_path, 'r') as f:
            test_img_ids = [line.strip() for line in f if line.strip()]
        return test_img_ids
    else:
        print(f"Test IDs file not found at {test_ids_path}. Re-splitting dataset to create test set.")
        return None

def create_test_set(dataset_dir, img_ext, model_dir, test_size=0.1, random_state=41):
    img_ids = glob(os.path.join(dataset_dir, 'images', '*' + img_ext))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
    
    # 以8:1:1的比例划分训练集、验证集和测试集
    train_img_ids, temp_img_ids = train_test_split(img_ids, test_size=0.2, random_state=random_state)
    val_img_ids, test_img_ids = train_test_split(temp_img_ids, test_size=0.5, random_state=random_state)
    
    # 确保模型目录存在
    os.makedirs(os.path.join('models', model_dir), exist_ok=True)
    
    # 保存测试集ID到文件
    test_ids_path = os.path.join('models', model_dir, 'test_ids.txt')
    with open(test_ids_path, 'w') as f:
        for img_id in test_img_ids:
            f.write(f"{img_id}\n")
    print(f"Test set IDs saved to {test_ids_path}")
    
    return test_img_ids

def visualize_multiclass_output(output, num_classes):
    """
    将多分类输出转换为彩色图像，每个类别使用不同的颜色。
    output: shape (C, H, W), 模型输出的 logits 值
    返回: shape (H, W, 3), 彩色图像
    """
    output = output.cpu().numpy()  # 获取 logits 值
    h, w = output.shape[-2:]
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 获取每个像素的预测类别
    pred_class = np.argmax(output, axis=0)  # shape (H, W)
    
    # 为每个类别分配颜色
    for c in range(num_classes):
        if c < len(COLOR_MAP):
            mask = pred_class == c
            color_img[mask] = COLOR_MAP[c]
    
    return color_img

def test(config, test_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    model.eval()
    results = []

    # 确保保存图片的目录存在
    output_dir = os.path.join('test_output', config.dataset, os.path.basename(config.model_dir), 'images')
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        pbar = tqdm(total=len(test_loader))
        for input, target, meta in test_loader:
            input = input.cuda()
            target = target.cuda()

            if config.deep_supervision:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou = iou_score(outputs[-1], target)
                output_to_save = outputs[-1]
            else:
                output = model(input)
                loss = criterion(output, target)
                iou = iou_score(output, target)
                output_to_save = output

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))

            # 收集每个样本的结果并保存推理图片
            for i in range(input.size(0)):
                img_id = meta['img_id'][i]
                results.append({
                    'img_id': img_id,
                    'loss': loss.item() if isinstance(loss, torch.Tensor) else loss,
                    'iou': iou if isinstance(iou, float) else iou.item()
                })
                
                # 保存推理结果图片（多分类彩色可视化）
                output_img = visualize_multiclass_output(output_to_save[i], config.num_classes)
                output_path = os.path.join(output_dir, f"{img_id}_pred.png")
                cv2.imwrite(output_path, cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)]), results

def main():
    config = parse_args()
    
    # 确保测试输出目录存在
    output_dir = os.path.join('test_output', config.dataset)
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载测试集ID
    test_img_ids = load_test_ids(config.model_dir, config.test_ids_file)
    if test_img_ids is None:
        dataset_dir = os.path.join('inputs', config.dataset)
        test_img_ids = create_test_set(dataset_dir, config.img_ext, config.model_dir)
    print(f"Loaded {len(test_img_ids)} test image IDs from {config.model_dir}")
    
    # 加载模型
    model_path = os.path.join('models', config.model_dir, 'model.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    print(f"Loading model from {model_path}")
    model = archs.__dict__[config.arch](config.num_classes,
                                        config.input_channels,
                                        # config.deep_supervision
    )
    model = model.cuda()
    model.load_state_dict(torch.load(model_path))
    
    # 定义损失函数
    if config.loss == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    elif config.loss == 'OneHotCrossEntropyLoss':
        criterion = losses.OneHotCrossEntropyLoss(ignore_index=config.num_classes - 1).cuda()
    elif config.loss == 'FocalLoss':
        criterion = losses.FocalLoss(alpha=1, gamma=1, ignore_index=config.num_classes - 1).cuda()
    else:
        criterion = losses.__dict__[config.loss]().cuda()
    
    # 准备数据
    test_transform = Compose([
        Resize(config.input_h, config.input_w),
        Normalize(),
    ])
    
    test_dataset = Dataset(
        img_ids=test_img_ids,
        img_dir=os.path.join('inputs', config.dataset, 'images'),
        mask_dir=os.path.join('inputs', config.dataset, 'masks'),
        img_ext=config.img_ext,
        mask_ext=config.mask_ext,
        num_classes=config.num_classes,
        transform=test_transform)
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=False)
    
    # 测试模型
    print("Starting testing on test set...")
    test_metrics, test_results = test(config, test_loader, model, criterion)
    print(f"Test metrics - Loss: {test_metrics['loss']:.4f}, IoU: {test_metrics['iou']:.4f}")
    
    # 保存测试结果
    results_df = pd.DataFrame(test_results)
    output_file = os.path.join(output_dir, f"{os.path.basename(config.model_dir)}_test_results.csv")
    results_df.to_csv(output_file, index=False)
    print(f"Test results saved to {output_file}")

if __name__ == '__main__':
    main()
