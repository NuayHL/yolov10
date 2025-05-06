from ultralytics import YOLO
#
model = YOLO('runs/detect/piou_4/weights/best.pt')

# metrics = model.val(data="ultralytics/cfg/datasets/coco.yaml", conf=0.001, iou=0.7, imgsz=640, batch=128, save_json=True)
metrics = model.val(data="ultralytics/cfg/datasets/coco.yaml", name='val_piou_4')
print(metrics)
model = YOLO('runs/detect1/InterpIoU-VisDrone-v8m-0dot03/weights/best.pt')

# metrics = model.val(data="ultralytics/cfg/datasets/coco.yaml", conf=0.001, iou=0.7, imgsz=640, batch=128, save_json=True)
metrics = model.val(data="ultralytics/cfg/datasets/VisDrone.yaml", name='val_interp_dot03_4')
print(metrics)