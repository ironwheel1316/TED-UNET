import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np # Not strictly needed for this version but good to have

# 定义已知的、你希望作为独立数据集标识符的模式
# 确保更长的模式（如果存在包含关系）放在前面，或者确保它们是互斥的
# 对于 "kvasir_seg_288" 和 "kvasir_seg_96"，顺序不影响，因为它们是不同的。
KNOWN_DATASET_PATTERNS = [
    "kvasir_seg_288",
    "kvasir_seg_96",
    # 在这里添加其他你想要明确区分的数据集标识符，例如:
    "dsb2018_96",
    "clinicdb_288",
    "clinicdb_96",
    # "another_dataset_specific"
]

def parse_directory_name(dirname):
    """
    解析目录名称以提取数据集标识符和模型名称。
    它会优先匹配 KNOWN_DATASET_PATTERNS 中的模式。
    """
    dataset_identifier = None
    model_name_part = "" # 初始化为空字符串

    # 尝试匹配已知的完整数据集模式
    for pattern in KNOWN_DATASET_PATTERNS:
        if dirname.startswith(pattern):
            # 确保是完整模式匹配，后面要么是字符串末尾，要么是下划线
            if len(dirname) == len(pattern) or dirname[len(pattern)] == '_':
                dataset_identifier = pattern
                model_name_part = dirname[len(pattern):]
                break
    
    if dataset_identifier is None:
        # 如果没有匹配到任何已知模式，则使用回退逻辑
        # 例如，可以尝试按第一个下划线分割
        parts = dirname.split('_', 1)
        if len(parts) > 1:
            dataset_identifier = parts[0]  # 例如 "dsb2018" 来自 "dsb2018_UNet"
            model_name_part = '_' + parts[1] # 保留下划线以便后续一致处理
        else:
            # 如果没有下划线，或者是非常短的名称，则归入一个通用类别
            dataset_identifier = "general_logs" # 或者 "unknown_dataset"
            model_name_part = dirname # 整个目录名作为模型部分
            
    # 清理模型名称部分（如果存在，则移除前导下划线）
    if model_name_part.startswith('_') and len(model_name_part) > 1:
        model_name_full = model_name_part[1:]
    else:
        model_name_full = model_name_part

    # 如果清理后的模型名称为空（例如，目录名恰好是 "kvasir_seg_288"）
    if not model_name_full and dataset_identifier:
        model_name_full = "model_for_" + dataset_identifier # 提供一个默认模型名
    elif not model_name_full and not dataset_identifier: # 理论上不应发生
        model_name_full = "unknown_model"
    elif not dataset_identifier: # 如果 dataset_identifier 也是空的 (理论上不应发生)
        dataset_identifier = "unknown_dataset_group"


    return dataset_identifier, model_name_full

def load_logs_by_dataset(log_dir): # 移除了 dataset_prefix_len 参数
    """
    加载日志并按数据集标识符（通过 parse_directory_name 解析）进行分组。
    """
    datasets_logs = {}
    if not os.path.exists(log_dir):
        print(f"日志目录 '{log_dir}' 不存在。")
        return datasets_logs

    for dirname in os.listdir(log_dir):
        dir_path = os.path.join(log_dir, dirname)
        if os.path.isdir(dir_path):
            log_file_path = os.path.join(dir_path, 'log.csv')
            if os.path.exists(log_file_path):
                # 使用新的解析函数
                dataset_identifier, model_name = parse_directory_name(dirname)

                if dataset_identifier not in datasets_logs:
                    datasets_logs[dataset_identifier] = []
                
                try:
                    log_data = pd.read_csv(log_file_path)
                    # 确保关键列是数字类型
                    for col in ['epoch', 'loss', 'iou', 'val_loss', 'val_iou', 'val_epoch']:
                        if col in log_data.columns:
                            log_data[col] = pd.to_numeric(log_data[col], errors='coerce')
                    
                    datasets_logs[dataset_identifier].append((model_name, log_data)) 
                except pd.errors.EmptyDataError:
                    print(f"警告: 空日志文件已跳过: {log_file_path}")
                except Exception as e:
                    print(f"警告: 加载日志文件 {log_file_path} 时出错: {e}")
            else:
                print(f"信息: 在 {dir_path} 中未找到 log.csv")
            
    return datasets_logs

# --- plot_comparison 和 print_summary_table 函数保持不变 ---
# (从你提供的代码中复制，它们应该能与新的 datasets_logs 结构一起工作)
def plot_comparison(logs, output_dir, dataset_name):
    if not logs:
        print(f"没有可为数据集 {dataset_name} 绘制的日志。")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6)) 
    
    colors = plt.cm.get_cmap('tab10', max(1,len(logs))).colors 
    linestyles_train = ['-', '--', '-.', ':'] * (max(1,len(logs)) // 4 + 1)
    linestyles_val = [':', '-.', '--', '-'] * (max(1,len(logs)) // 4 + 1)

    ax1 = axes[0]
    for i, (model_name, log_data) in enumerate(logs):
        color = colors[i % len(colors)]
        ax1.plot(log_data['epoch'].dropna(), log_data['loss'].dropna(), 
                 label=f'{model_name} Train Loss', color=color, linestyle=linestyles_train[i])
        if 'val_loss' in log_data.columns and 'val_epoch' in log_data.columns:
            actual_val_points = log_data[
                (log_data['epoch'] == log_data['val_epoch']) & log_data['val_loss'].notna()
            ].copy()
            if not actual_val_points.empty:
                ax1.plot(actual_val_points['epoch'], actual_val_points['val_loss'], 
                         label=f'{model_name} Val Loss', color=color, linestyle=linestyles_val[i], marker='.')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Loss Comparison - Dataset: {dataset_name}')
    ax1.grid(True)
    
    ax2 = axes[1]
    for i, (model_name, log_data) in enumerate(logs):
        color = colors[i % len(colors)]
        ax2.plot(log_data['epoch'].dropna(), log_data['iou'].dropna(), 
                 label=f'{model_name} Train IoU', color=color, linestyle=linestyles_train[i])
        if 'val_iou' in log_data.columns and 'val_epoch' in log_data.columns:
            actual_val_points = log_data[
                (log_data['epoch'] == log_data['val_epoch']) & log_data['val_iou'].notna()
            ].copy()
            if not actual_val_points.empty:
                ax2.plot(actual_val_points['epoch'], actual_val_points['val_iou'], 
                         label=f'{model_name} Val IoU', color=color, linestyle=linestyles_val[i], marker='.')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('IoU')
    ax2.set_title(f'IoU Comparison - Dataset: {dataset_name}')
    ax2.grid(True)
    
    handles, labels = [], []
    for ax in fig.axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    
    by_label = dict(zip(labels, handles))
    
    num_legend_items = len(by_label)
    ncol_legend = 1
    if num_legend_items > 10: 
        ncol_legend = 2
    if num_legend_items > 20: 
        ncol_legend = 3
    
    fig.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', ncol=ncol_legend)
    
    plt.tight_layout(rect=[0, 0, 0.85, 0.96]) 
    fig.suptitle(f'Model Performance Comparison for Dataset: {dataset_name}', fontsize=14)
    
    line_plot_path = os.path.join(output_dir, f'comparison_plot_lines_{dataset_name}.png')
    plt.savefig(line_plot_path, bbox_inches='tight') 
    plt.close(fig) 
    print(f"折线对比图 {dataset_name} 已保存至 {line_plot_path}")

    best_val_ious_data = {}
    for model_name, log_data in logs:
        if 'val_iou' in log_data.columns and 'val_epoch' in log_data.columns:
            actual_val_points = log_data[
                (log_data['epoch'] == log_data['val_epoch']) & log_data['val_iou'].notna()
            ].copy()
            if not actual_val_points.empty:
                best_val_ious_data[model_name] = actual_val_points['val_iou'].max()
            else:
                best_val_ious_data[model_name] = 0 
        else:
            best_val_ious_data[model_name] = 0

    if best_val_ious_data:
        sorted_models = sorted(best_val_ious_data.keys(), key=lambda m: best_val_ious_data[m], reverse=True)
        sorted_ious = [best_val_ious_data[m] for m in sorted_models]

        plt.figure(figsize=(max(6, len(sorted_models) * 0.9), 6))
        bars = plt.bar(sorted_models, sorted_ious, color=colors[:len(sorted_models)])
        plt.xlabel('Model')
        plt.ylabel('Best Validation IoU')
        plt.title(f'Best Validation IoU - Dataset: {dataset_name}')
        plt.xticks(rotation=45, ha="right", fontsize='small')
        plt.yticks(fontsize='small')
        plt.ylim(bottom=max(0, min(sorted_ious)-0.05 if sorted_ious else 0), top=min(1, max(sorted_ious)+0.05 if sorted_ious else 1))
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.005, f'{yval:.4f}', ha='center', va='bottom', fontsize='x-small')

        plt.tight_layout()
        bar_plot_path = os.path.join(output_dir, f'best_val_iou_bar_{dataset_name}.png')
        plt.savefig(bar_plot_path)
        plt.close()
        print(f"最佳验证集IoU柱状图 {dataset_name} 已保存至 {bar_plot_path}")

def print_summary_table(logs, output_dir, dataset_name):
    summary_data = []
    for model_name, log_data in logs:
        model_summary = {'Model': model_name}
        
        has_val_data = 'val_iou' in log_data.columns and \
                       'val_loss' in log_data.columns and \
                       'val_epoch' in log_data.columns

        if has_val_data:
            actual_val_points = log_data[
                (log_data['epoch'] == log_data['val_epoch'])
            ].copy()

            if not actual_val_points.empty:
                valid_iou_points = actual_val_points['val_iou'].dropna()
                if not valid_iou_points.empty:
                    best_iou = valid_iou_points.max()
                    best_iou_epoch = actual_val_points.loc[valid_iou_points.idxmax(), 'epoch']
                    model_summary['Best Val IoU'] = f"{best_iou:.4f} (E{int(best_iou_epoch)})"
                else:
                    model_summary['Best Val IoU'] = "N/A"
                
                final_iou_series = actual_val_points['val_iou'].dropna()
                if not final_iou_series.empty:
                    final_iou = final_iou_series.iloc[-1]
                    final_iou_epoch = actual_val_points.loc[final_iou_series.index[-1], 'epoch']
                    model_summary['Final Val IoU'] = f"{final_iou:.4f} (E{int(final_iou_epoch)})"
                else:
                    model_summary['Final Val IoU'] = "N/A"

                valid_loss_points = actual_val_points['val_loss'].dropna()
                if not valid_loss_points.empty:
                    best_loss = valid_loss_points.min()
                    best_loss_epoch = actual_val_points.loc[valid_loss_points.idxmin(), 'epoch']
                    model_summary['Best Val Loss'] = f"{best_loss:.4f} (E{int(best_loss_epoch)})"
                else:
                    model_summary['Best Val Loss'] = "N/A"

                final_loss_series = actual_val_points['val_loss'].dropna()
                if not final_loss_series.empty:
                    final_loss = final_loss_series.iloc[-1]
                    final_loss_epoch = actual_val_points.loc[final_loss_series.index[-1], 'epoch']
                    model_summary['Final Val Loss'] = f"{final_loss:.4f} (E{int(final_loss_epoch)})"
                else:
                    model_summary['Final Val Loss'] = "N/A"
            else:
                model_summary.update({'Best Val IoU': "No actual val points", 'Final Val IoU': "No actual val points",
                                      'Best Val Loss': "No actual val points", 'Final Val Loss': "No actual val points"})
        else:
            model_summary.update({'Best Val IoU': "No val data", 'Final Val IoU': "No val data",
                                  'Best Val Loss': "No val data", 'Final Val Loss': "No val data"})
            
        summary_data.append(model_summary)

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.set_index('Model', inplace=True)
        print(f"\n--- 数据集性能总结: {dataset_name} ---")
        try:
            from tabulate import tabulate
            print(tabulate(summary_df, headers='keys', tablefmt='psql'))
        except ImportError:
            print(summary_df.to_string())
        
        summary_table_path = os.path.join(output_dir, f'performance_summary_{dataset_name}.csv')
        summary_df.to_csv(summary_table_path)
        print(f"总结表格 {dataset_name} 已保存至 {summary_table_path}")
    else:
        print(f"没有为数据集 {dataset_name} 生成的总结数据。")

def main():
    log_dir = 'models'
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    # dataset_prefix_length 不再用于核心解析逻辑，可以移除或忽略
    all_datasets_logs = load_logs_by_dataset(log_dir) # 调用时不传递 prefix_len
    
    if not all_datasets_logs:
        print(f"在 '{log_dir}' 中未找到日志目录，或日志无法正确解析。") # 更新了提示信息
        return
    
    for dataset_identifier, logs_for_dataset in all_datasets_logs.items():
        if logs_for_dataset:
            print(f"\n正在处理数据集组: {dataset_identifier}")
            plot_comparison(logs_for_dataset, output_dir, dataset_identifier)
            print_summary_table(logs_for_dataset, output_dir, dataset_identifier)
        else:
            print(f"未找到或解析数据集组的日志: {dataset_identifier}")

if __name__ == '__main__':
    main()