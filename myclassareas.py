import os
import yaml
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import numpy as np


def plot_class_box_sizes(yaml_path, save_path='box_sizes.png'):
    # 1. 读取 YAML 文件
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    base_path = data['path']
    test_dirs = data['test']
    class_names = data['names']

    if isinstance(test_dirs, str):
        test_dirs = [test_dirs]

    label_dirs = [os.path.join(base_path, p.replace('images', 'labels')) for p in test_dirs]

    # 2. 收集所有 txt 文件路径
    label_files = []
    for label_dir in label_dirs:
        if not os.path.exists(label_dir):
            print(f"Warning: label directory {label_dir} does not exist.")
            continue
        label_files.extend([
            os.path.join(label_dir, f) for f in os.listdir(label_dir)
            if f.endswith('.txt')
        ])

    # 3. 读取 label 数据，统计每类目标框面积
    class_box_sizes = defaultdict(list)
    for label_path in tqdm(label_files, desc="Processing label files"):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id = int(parts[0])
                width = float(parts[3])
                height = float(parts[4])
                area = width * height
                class_box_sizes[cls_id].append(area)

    # 4. 准备数据绘图
    labels = []
    data = []
    for cls_id in sorted(class_box_sizes.keys()):
        labels.append(class_names[cls_id])
        data.append(class_box_sizes[cls_id])

    plt.figure(figsize=(12, 3.5))  # 宽不变，高度更紧凑
    plt.boxplot(
        data,
        labels=labels,
        showfliers=False,
        whis=[5, 75],  # 限制须长度
        patch_artist=True,
        boxprops=dict(facecolor='#91c9f7', color='#2878B5', linewidth=1.5),
        medianprops=dict(color='#D14A61', linewidth=2),
        whiskerprops=dict(color='#2878B5', linewidth=1.5),
        capprops=dict(color='#2878B5', linewidth=1.5)
    )
    plt.xticks(rotation=90, fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel("Normalized Box Area", fontsize=15)
    plt.title("Box Size Distribution per Class in VisDrone", fontsize=16, fontweight='bold')
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout(pad=0.3)  # 更紧凑布局
    plt.savefig(save_path, dpi=300, bbox_inches='tight')  # 高分辨率输出 + 紧边距
    plt.close()
    print(f"📦 Box plot saved to: {save_path}")

def collect_box_sizes(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    base_path = data['path']
    test_dirs = data['test']
    class_names = data['names']

    if isinstance(test_dirs, str):
        test_dirs = [test_dirs]
    label_dirs = [os.path.join(base_path, p.replace('images', 'labels')) for p in test_dirs]

    label_files = []
    for label_dir in label_dirs:
        if not os.path.exists(label_dir):
            print(f"Warning: label directory {label_dir} does not exist.")
            continue
        label_files.extend([
            os.path.join(label_dir, f) for f in os.listdir(label_dir)
            if f.endswith('.txt')
        ])

    class_box_sizes = defaultdict(list)
    for label_path in tqdm(label_files, desc=f"Processing {os.path.basename(yaml_path)}"):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id = int(parts[0])
                width = float(parts[3])
                height = float(parts[4])
                area = width * height
                class_box_sizes[cls_id].append(area)

    return class_box_sizes, class_names


def plot_two_datasets_boxplot(yaml1, yaml2, title1='VOC', title2='VisDrone', save_path='datasets_boxplot.png'):
    # 收集两个数据集的框面积
    box1, names1 = collect_box_sizes(yaml1)
    box2, names2 = collect_box_sizes(yaml2)

    # 过滤中位数大于 0.1 的类
    def filter_by_median(box_dict):
        return {
            k: v for k, v in box_dict.items()
            if len(v) > 0 and np.median(v) <= 0.1
        }

    box1_filtered = filter_by_median(box1)
    box2_filtered = filter_by_median(box2)

    # 准备图像
    # fig, axes = plt.subplots(2, 1, figsize=(6, 8), sharex=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3))

    def plot_subplot(ax, box_dict, names, title):
        class_ids = sorted(box_dict.keys())
        data = [box_dict[i] for i in class_ids]
        labels = [names[i] for i in class_ids]

        # ax.boxplot(data, labels=labels, showfliers=False)
        ax.boxplot(
            data,
            labels=labels,
            showfliers=False,
            whis=[5, 75],  # 可选：限制须长度在 5%-95%
            patch_artist=True,
            boxprops=dict(facecolor='#91c9f7', color='#2878B5'),
            medianprops=dict(color='#D14A61'),
            whiskerprops=dict(color='#2878B5'),
            capprops=dict(color='#2878B5')
        )
        ax.set_title(title)
        ax.set_ylabel("Normalized Area",fontsize=8)
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        ax.grid(True)

    plot_subplot(axes[0], box1_filtered, names1, title1)
    plot_subplot(axes[1], box2_filtered, names2, title2)

    plt.tight_layout(pad=0.2)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📄 Combined boxplot saved to {save_path}")


if __name__ == "__main__":
    plot_class_box_sizes("ultralytics/cfg/datasets/VOC.yaml", save_path="voc_test_box_sizes.png")
    plot_class_box_sizes("ultralytics/cfg/datasets/VisDrone.yaml", save_path="visdrone_test_box_sizes.png")
    # plot_two_datasets_boxplot(
    #     yaml1="ultralytics/cfg/datasets/VisDrone.yaml",
    #     yaml2="ultralytics/cfg/datasets/VOC.yaml",
    #     title2="VOC (Test Set)",
    #     title1="VisDrone (Test Set)",
    #     save_path="voc_visdrone_boxplot.png"
    # )
