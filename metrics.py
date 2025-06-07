import numpy as np
import torch
import torch.nn.functional as F

# 计算 IoU（Intersection over Union，交并比），支持多分类
def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    
    # 获取类别数
    num_classes = output.shape[1] if len(output.shape) > 3 else 1
    
    if num_classes == 1:
        output_ = output > 0.5
        target_ = target > 0.5
        intersection = (output_ & target_).sum()  # 交集
        union = (output_ | target_).sum()  # 并集
        return (intersection + smooth) / (union + smooth)
    else:
        # 多分类情况，计算每个类别的 IoU 并取平均值
        iou_per_class = []
        # 获取预测类别图
        pred_class = np.argmax(output, axis=1)  # shape (N, H, W)
        target_class = np.argmax(target, axis=1)  # shape (N, H, W)
        for c in range(num_classes-1):
            output_c = pred_class == c
            target_c = target_class == c
            intersection = (output_c & target_c).sum()  # 交集
            union = (output_c | target_c).sum()  # 并集
            if union == 0:  # 避免除以零
                continue
            iou_per_class.append((intersection + smooth) / (union + smooth))
        return np.mean(iou_per_class) if iou_per_class else 0.0

#Dice 系数:衡量预测与目标的重叠区域占两者总和的比例。
#更关注预测与目标的重叠部分。
def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)
