import argparse

#在命令行参数解析时，将类似 'true' 或 'false' 的字符串转换为布尔值
def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

#统计模型中需要训练的参数数量
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#用于计算和存储训练过程中某个指标的当前值、总和、计数和平均值
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
