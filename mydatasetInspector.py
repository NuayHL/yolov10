import os
import cv2 as cv
import numpy as np
import yaml
import os
from ultralytics.utils.plotting import Annotator, colors
from ultralytics import YOLO

dataset_cfg = 'ultralytics/cfg/datasets/VisDrone.yaml'
using_type = 'val'
output_dir = 'comparison_no_label_new'

os.makedirs(output_dir, exist_ok=True)

with open(dataset_cfg, 'r') as f:
    dataset_cfg = yaml.safe_load(f)

names = dataset_cfg['names']

dataset_dir = dataset_cfg['path']
image_dir = str(os.path.join(dataset_dir, dataset_cfg[using_type]))
label_dir = str(os.path.join(os.path.dirname(image_dir),'labels'))

for img_file in os.listdir(image_dir):
    if img_file.endswith('.jpg') or img_file.endswith('.png'):
        img_path = os.path.join(image_dir, img_file)
        label_path = os.path.join(label_dir, img_file.replace('.jpg', '.txt').replace('.png', '.txt'))

        image = cv.imread(img_path)
        h_img, w_img = image.shape[:2]

        ann = Annotator(
            image,
            line_width=2,  # 线宽
            font_size=14,  # 字体大小
            font="Arial.ttf",  # 使用 ImageFont 兼容字体
            pil=False,  # 使用 OpenCV 绘制
        )

        # 读取标签文件
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                labels = [line.strip().split() for line in f.readlines()]

            # 解析 YOLO 格式的标签
            for nb, label in enumerate(labels):
                class_id, x_center, y_center, w, h = map(float, label)
                x1 = int((x_center - w / 2) * w_img)
                y1 = int((y_center - h / 2) * h_img)
                x2 = int((x_center + w / 2) * w_img)
                y2 = int((y_center + h / 2) * h_img)

                class_name = names.get(int(class_id), "Unknown")
                label_text = f"{str(nb).zfill(2)}: {class_name}"

                # ann.box_label([x1, y1, x2, y2], label_text, color=colors(int(class_id), bgr=True))
                ann.box_label([x1, y1, x2, y2], '', color=colors(int(class_id), bgr=True))

        image_with_bboxes = ann.result()

        output_path = os.path.join(output_dir, img_file).replace('.jpg', '_gt.jpg')
        cv.imwrite(output_path, image_with_bboxes)

print(f"All images with gt has been saved in：{output_dir}")

