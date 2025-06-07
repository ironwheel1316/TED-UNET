import torch
from archs import NOR1UNet444,VGGBlock

def track_dimensions(model, model_name, input_size=(3, 96, 96)):
    """
    追踪并打印模型中每一层的特征图尺寸和通道数变化。
    参数:
        model: 模型实例
        model_name: 模型名称
        input_size: 输入图像尺寸，格式为 (通道数, 高度, 宽度)
    """
    print(f"\n=== {model_name} 模型尺寸和通道数变化 ===")
    dummy_input = torch.randn(1, *input_size)
    current_size = input_size[1:]  # 仅保留高度和宽度
    current_channels = input_size[0]  # 当前通道数
    
    # 钩子函数，用于捕获每一层的输出尺寸和通道数
    def hook_fn(module, input, output, name):
        nonlocal current_size, current_channels
        if isinstance(module, torch.nn.MaxPool2d):
            current_size = (current_size[0] // 2, current_size[1] // 2)
            print(f"{name} (MaxPool2d): 尺寸={current_size}, 通道数={current_channels}")
        elif isinstance(module, torch.nn.Upsample):
            current_size = (current_size[0] * 2, current_size[1] * 2)
            print(f"{name} (Upsample): 尺寸={current_size}, 通道数={current_channels}")
        elif isinstance(module, torch.nn.Conv2d):
            current_channels = module.out_channels
            print(f"{name} (Conv2d): 尺寸={current_size}, 通道数={current_channels}")
        elif 'VGGBlock' in str(type(module)):
            if hasattr(module, 'conv2'):
                current_channels = module.conv2.out_channels
            print(f"{name} (VGGBlock): 尺寸={current_size}, 通道数={current_channels}")
        elif hasattr(module, 'forward') and 'cat' in str(module.forward.__code__):
            # 对于连接操作，通道数是连接后的总和，尺寸不变
            if isinstance(output, list):
                current_channels = sum(o.shape[1] for o in output if isinstance(o, torch.Tensor))
            else:
                current_channels = output.shape[1]
            print(f"{name} (Concat): 尺寸={current_size}, 通道数={current_channels}")
        else:
            if isinstance(output, torch.Tensor):
                current_channels = output.shape[1]
            print(f"{name}: 尺寸={current_size}, 通道数={current_channels}")

    # 注册钩子到模型的每一层
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.MaxPool2d, torch.nn.Upsample, torch.nn.Conv2d, VGGBlock)) or 'conv' in name.lower():
            hooks.append(module.register_forward_hook(lambda m, i, o, n=name: hook_fn(m, i, o, n)))

    # 前向传播以触发钩子
    with torch.no_grad():
        model(dummy_input)

    # 移除钩子
    for hook in hooks:
        hook.remove()

def main():
    # 定义输入尺寸
    input_size = (3, 96, 96)  # 通道数, 高度, 宽度
    
    # 实例化所有模型
    models = {
        "NOR1UNet444": NOR1UNet444(num_classes=1),  # 提供 num_classes 参数
    }
    
    # 对每个模型追踪尺寸变化
    for name, model in models.items():
        track_dimensions(model, name, input_size)

if __name__ == "__main__":
    main()
