import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import time

def extract_arch_from_model_name(model_name):
    """
    从模型名称中提取架构名称。
    假设模型名称格式为：<prefix>_<arch>_<suffix>
    """
    parts = model_name.split("_")
    if len(parts) == 7:
        return parts[3]  # 第三个部分是架构名称
    elif len(parts) == 6:
        return parts[2]
    return None

def test_models(models_dir, prefix=None, dataset=None, input_h=96, input_w=96):
    """
    测试所有模型或指定前缀的模型，并保存结果到 CSV 文件。
    """
    results = []
    for model_dir in os.listdir(models_dir):
        # 如果未指定前缀，则测试所有模型；如果指定了前缀，则只测试匹配的模型
        if prefix is None or model_dir.startswith(prefix):
            print(f"Testing model: {model_dir}")
            arch = extract_arch_from_model_name(model_dir)
            if not arch:
                print(f"Error: Unable to extract architecture from model name '{model_dir}'. Skipping...")
                continue
            command = [
                "python", "test.py",
                "--model_dir", model_dir,
                "--dataset", dataset,
                "--arch", arch,
                "--input_h", str(input_h),
                "--input_w", str(input_w)
            ]
            try:
                # 执行测试命令
                output = subprocess.check_output(command, stderr=subprocess.STDOUT, universal_newlines=True, encoding='utf-8')
                print(output)
                # 从输出中提取 IoU 和 Loss
                for line in output.splitlines():
                    if "Test metrics" in line:
                        metrics = line.split("-")[1].strip()
                        loss, iou = metrics.split(", ")
                        loss = float(loss.split(":")[1])
                        iou = float(iou.split(":")[1])
                        results.append({"model": model_dir, "arch": arch, "loss": loss, "iou": iou})
            except subprocess.CalledProcessError as e:
                print(f"Error testing model {model_dir}: {e.output}")
    return results

def plot_results(results, output_file="comparison_plot.png"):
    """
    绘制模型架构的 IoU、Dice 和 Loss 的比较图（设置 IoU 和 Dice 起始值）。
    """
    if len(results) == 0:
        print("No results to plot.")
        return

    # 转换为 DataFrame
    df = pd.DataFrame(results)
    df = df.sort_values(by="iou", ascending=False)  # 按 IoU 排序

    # 添加 Dice 列（通过 IoU 转换）
    df["dice"] = 2 * df["iou"] / (1 + df["iou"])

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 绘制 IoU 的柱状图
    bar_width = 0.4  # 柱状图宽度
    x = range(len(df["arch"]))
    ax1.bar(x, df["iou"] * 100, width=bar_width, label="IoU (%)", alpha=0.7, color="blue", align="center")

    # 绘制 Dice 的柱状图
    ax1.bar([i + bar_width for i in x], df["dice"] * 100, width=bar_width, label="Dice (%)", alpha=0.7, color="green", align="center")

    ax1.set_ylabel("Percentage (%)", color="black")
    ax1.tick_params(axis="y", labelcolor="black")
    ax1.set_xlabel("Architecture")
    ax1.set_xticks([i + bar_width / 2 for i in x])
    ax1.set_xticklabels(df["arch"], rotation=45, ha="right")
    ax1.set_ylim(50, 100)  # 设置 Y 轴起始值为 50%

    # 在柱状图上添加数据标签（IoU 和 Dice）
    for i, (iou, dice) in enumerate(zip(df["iou"] * 100, df["dice"] * 100)):
        ax1.text(i, iou + 1, f"{iou:.1f}%", ha="center", color="blue", fontsize=9)
        ax1.text(i + bar_width, dice + 1, f"{dice:.1f}%", ha="center", color="green", fontsize=9)

    # 绘制 Loss 的折线图（使用第二个 Y 轴）
    ax2 = ax1.twinx()
    ax2.plot(df["arch"], df["loss"], label="Loss", color="red", marker="o", linestyle="--")
    ax2.set_ylabel("Loss", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    # 在 Loss 折线图上添加数据标签
    for i, v in enumerate(df["loss"]):
        ax2.text(i, v + 0.01, f"{v:.2f}", ha="center", color="red", fontsize=9)

    # 添加标题和图例
    fig.suptitle("Model Comparison (IoU, Dice, and Loss)", fontsize=14)
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Comparison plot saved to {output_file}")

def main():
    models_dir = "D:\\pytorch-nested-unet\\models"
    prefix = input("Enter model prefix (or press Enter to test all models): ").strip()
    prefix = prefix if prefix else None  # 如果用户未输入前缀，则设置为 None
    dataset = input("Enter dataset name: ").strip()
    
    results = test_models(models_dir, prefix, dataset)
    if results:
        results_file = os.path.join(models_dir, f"{prefix or 'all'}_test_results.csv")
        pd.DataFrame(results).to_csv(results_file, index=False)
        print(f"Test results saved to {results_file}")
        plot_results(results, output_file=os.path.join(models_dir, f"{prefix or 'all'}_comparison_plot.png"))
    else:
        print("No models found with the specified prefix.")

if __name__ == "__main__":
    main()
    #kvasir_seg_96