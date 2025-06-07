import os
import torch
from torchvision.models import resnet18  # 示例模型
from torch.profiler import profile, ProfilerActivity
from ptflops import get_model_complexity_info  # 使用 ptflops 库计算 FLOPS
import archs

def calculate_memory_access(model, input_size):
    """
    使用 torch.profiler 计算模型的访存量。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dummy_input = torch.randn(1, *input_size).to(device)  # 创建一个虚拟输入

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        model(dummy_input)  # 执行一次前向传播

    # 提取访存量信息
    memory_stats = prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10)
    print(memory_stats)
    return memory_stats

def extract_arch_from_model_name(model_name):
    """
    从模型名称中提取架构名称。
    假设模型名称格式为：<prefix>_<arch>_<suffix>
    """
    parts = model_name.split("_")
    if len(parts) >= 4:
        return parts[2]  # 第三个部分是架构名称
    return None

def calculate_flops(models_dir, prefix=None, input_size=(3, 224, 224)):
    """
    计算前缀为某特定的所有模型的 FLOPS。
    """
    results = []
    for model_dir in os.listdir(models_dir):
        # 过滤非模型文件
        if not os.path.isdir(os.path.join(models_dir, model_dir)):
            continue
        if prefix is None or model_dir.startswith(prefix):
            print(f"Calculating FLOPS for model: {model_dir}")
            arch = extract_arch_from_model_name(model_dir)
            if not arch:
                print(f"Error: Unable to extract architecture from model name '{model_dir}'. Skipping...")
                continue
            
            # 加载模型（根据架构名称加载对应的模型）
            try:
                if arch == "UNet":
                    model = archs.UNet(num_classes=1)  # 提供 num_classes 参数
                elif arch == "NestedUNet":
                    model = archs.NestedUNet(num_classes=1)  # 提供 num_classes 参数
                elif arch == "UNetPlusMinus":
                    model = archs.UNetPlusMinus(num_classes=1)  # 提供 num_classes 参数
                elif arch == "UNetPlusMinus2":
                    model = archs.UNetPlusMinus2(num_classes=1)  # 提供 num_classes 参数
                elif arch == "WNet":
                    model = archs.WNet(num_classes=1)  # 提供 num_classes 参数
                elif arch == "UNetR1":
                    model = archs.UNetR1(num_classes=1)  # 提供 num_classes 参数
                elif arch == "UNetR12":
                    model = archs.UNetR12(num_classes=1)  # 提供 num_classes 参数
                elif arch == "WNetPlusPlus":
                    model = archs.WNetPlusPlus(num_classes=1)  # 提供 num_classes 参数
                elif arch == "UNetDeep":
                    model = archs.UNetDeep(num_classes=1)  # 提供 num_classes 参数
                elif arch == "Conv3UNet":
                    model = archs.Conv3UNet(num_classes=1)  # 提供 num_classes 参数
                elif arch == "ConvUNet123":
                    model = archs.ConvUNet123(num_classes=1)  # 提供 num_classes 参数
                elif arch == "ConvUNet321":
                    model = archs.ConvUNet321(num_classes=1)  # 提供 num_classes 参数
                elif arch == "ConvUNet31122":
                    model = archs.ConvUNet31122(num_classes=1)  # 提供 num_classes 参数
                elif arch == "ConvUNet32222":
                    model = archs.ConvUNet32222(num_classes=1)  # 提供 num_classes 参数
                elif arch == "ConvUNet32211":
                    model = archs.ConvUNet32211(num_classes=1)  # 提供 num_classes 参数
                elif arch == "ConvUNet3333":
                    model = archs.ConvUNet3333(num_classes=1)  # 提供 num_classes 参数
                elif arch == "ConvUNet444":
                    model = archs.ConvUNet444(num_classes=1)  # 提供 num_classes 参数
                elif arch == "ConvUNet55":
                    model = archs.ConvUNet55(num_classes=1)  # 提供 num_classes 参数
                elif arch == "ConvUNet6":
                    model = archs.ConvUNet6(num_classes=1)  # 提供 num_classes 参数
                elif arch == "R1UNet444":
                    model = archs.R1UNet444(num_classes=1)  # 提供 num_classes 参数
                else:
                    print(f"Unsupported architecture: {arch}. Skipping...")
                    continue
                
                # 使用 ptflops 计算 FLOPS
                flops, params = get_model_complexity_info(model, input_size, as_strings=True, print_per_layer_stat=False)
                memory_stats = calculate_memory_access(model, input_size)
                print(f"Model: {model_dir}, FLOPS: {flops}, Params: {params}, Memory Access: {memory_stats}")
                results.append({"model": model_dir, "arch": arch, "flops": flops, "params": params, "memory_access": memory_stats})
            except Exception as e:
                print(f"Error calculating FLOPS for model {model_dir}: {e}")
    return results

def save_results(results, output_file="flops_results.csv"):
    """
    保存 FLOPS 计算结果到 CSV 文件。
    """
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

def main():
    models_dir = "D:\\pytorch-nested-unet\\models"
    prefix = input("Enter model prefix (or press Enter to calculate for all models): ").strip()
    prefix = prefix if prefix else None  # 如果用户未输入前缀，则设置为 None
    input_size = (3, 96, 96)  # 输入大小（根据模型需求调整）

    results = calculate_flops(models_dir, prefix, input_size)
    if results:
        save_results(results, output_file=os.path.join(models_dir, f"{prefix or 'all'}_flops_results.csv"))
    else:
        print("No models found with the specified prefix.")

if __name__ == "__main__":
    main()