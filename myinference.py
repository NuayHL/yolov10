import cv2 as cv
import yaml
import os
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from tqdm import tqdm

def process_images(model_pt, model_suffix, dataset_cfg, using_type, output_dir):
    """
    处理图像，绘制边界框，并将结果保存在指定目录中。

    参数：
    - model_pt (str): 模型权重的路径
    - model_suffix (str): 输出图像的后缀
    - dataset_cfg (str): 数据集的配置文件路径
    - using_type (str): 使用的数据集类型 (如 'val' 或 'test')
    - output_dir (str): 处理后图像的输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 读取数据集配置
    with open(dataset_cfg, 'r') as f:
        dataset_cfg = yaml.safe_load(f)

    names = dataset_cfg['names']
    dataset_dir = dataset_cfg['path']
    image_dir = str(os.path.join(dataset_dir, dataset_cfg[using_type]))

    # 读取图像列表
    image_list = os.listdir(image_dir)

    # 加载 YOLO 模型
    model = YOLO(model_pt)

    # 处理图像
    for image_name in tqdm(image_list, desc="Processing Images", unit="img"):
        image_path = os.path.join(image_dir, image_name)
        image = cv.imread(image_path)

        # 获取预测结果
        results = model(image)[0]

        # 初始化 Annotator
        ann = Annotator(
            image,
            line_width=2,  # 线宽
            font_size=14,  # 字体大小
            font="Arial.ttf",  # 使用 ImageFont 兼容字体
            pil=False,  # 使用 OpenCV 绘制
        )

        names = results.names
        for d in reversed(results.boxes):
            c, conf, id = int(d.cls), float(d.conf), None if d.id is None else int(d.id.item())
            name = ("" if id is None else f"id:{id} ") + names[c]
            label = (f"{name} {conf:.2f}" if conf else name)
            box = d.xyxy.squeeze()

            # 绘制边界框，不显示标签
            ann.box_label(box, '', color=colors(c, True), rotated=False)

        # 获取绘制后的图像
        image_with_bboxes = ann.result()

        # 生成新文件名并保存
        pre_img_name = image_name.replace('.jpg', f'{model_suffix}.jpg')
        cv.imwrite(os.path.join(output_dir, pre_img_name), image_with_bboxes)


if __name__ == "__main__":
    # process_images(
    #     model_pt="runs/detect1/InterpIoU-VisDrone-v8m-0dot2/weights/best.pt",
    #     model_suffix="_interp",
    #     dataset_cfg="ultralytics/cfg/datasets/VisDrone.yaml",
    #     using_type="val",
    #     output_dir="comparison_no_label"
    # )
    process_images(
        model_pt="runs/detect/vis_re/vis_60_99/weights/best.pt",
        model_suffix="_interpdy",
        dataset_cfg="ultralytics/cfg/datasets/VisDrone.yaml",
        using_type="val",
        output_dir="comparison_no_label_new"
    )
    # process_images(
    #     model_pt="runs/detect/vis_re/piou_vis_re/weights/best.pt",
    #     model_suffix="_piou",
    #     dataset_cfg="ultralytics/cfg/datasets/VisDrone.yaml",
    #     using_type="val",
    #     output_dir="comparison_no_label_new"
    # )

