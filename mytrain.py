from ultralytics import YOLO

model = YOLO('runs/detect/piou/weights/best.pt')

# metrics = model.val(data="ultralytics/cfg/datasets/coco.yaml", conf=0.001, iou=0.7, imgsz=640, batch=128, save_json=True)
metrics = model.val(data="ultralytics/cfg/datasets/coco.yaml", imgsz=640, batch=32, name='val_piou')
# print(metrics)
