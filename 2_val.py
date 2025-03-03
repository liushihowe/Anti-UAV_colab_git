# 模型验证
from ultralytics import YOLO

import torch
# 自动选择设备，GPU 优先，如果没有 GPU，则使用 CPU
device = torch.device(0 if torch.cuda。is_available() else 'cpu')

# Load a model

model = YOLO("./runs/train_yoloi.yaml_2025-02-23_11-36-57/weights/best.pt")
# model = YOLO("./runs/train_yolon.yaml_2025-03-03_08-03-43/weights/best.pt")

# Customize validation settings
validation_results = model.val(data="./datasets/data_val.yaml", imgsz=320, batch=64, conf=0.25, iou=0.6, device=device, project = "runs", name="val_yoloi")

print(validation_results.box。map)  # mAP50-95
print(validation_results.box。map50)  # mAP50
print(validation_results.box。map75)  # mAP75
print(validation_results.box。maps)  # list of mAP50-95 for each category
