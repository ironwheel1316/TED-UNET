import os
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm

# 该脚本用于将训练的数据集转换为指定格式，并放于dsb2018_%d
# 将原图转换为三通道，mask处理（合并掩码）
# 注意每次处理会覆盖前一次处理好的（size不同可保留）


def main():
    img_size = 96

    paths = glob('inputs/data-science-bowl-2018/stage1_train/*')

    os.makedirs('inputs/dsb2018_%d/images' % img_size, exist_ok=True)
    os.makedirs('inputs/dsb2018_%d/masks/0' % img_size, exist_ok=True)

    for i in tqdm(range(len(paths))):
        path = paths[i]
        img = cv2.imread(os.path.join(path, 'images',
                         os.path.basename(path) + '.png'))
        mask = np.zeros((img.shape[0], img.shape[1]))
        for mask_path in glob(os.path.join(path, 'masks', '*')):
            mask_ = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 127
            mask[mask_] = 1#将掩码图像的像素值阈值化（大于 127 的像素值设置为 True，否则为 False）
        if len(img.shape) == 2:
            img = np.tile(img[..., None], (1, 1, 3))
        if img.shape[2] == 4:
            img = img[..., :3]#原图强行三通道
        img = cv2.resize(img, (img_size, img_size))
        mask = cv2.resize(mask, (img_size, img_size))
        cv2.imwrite(os.path.join('inputs/dsb2018_%d/images' % img_size,
                    os.path.basename(path) + '.png'), img)
        cv2.imwrite(os.path.join('inputs/dsb2018_%d/masks/0' % img_size,
                    os.path.basename(path) + '.png'), (mask * 255).astype('uint8'))


if __name__ == '__main__':
    main()

