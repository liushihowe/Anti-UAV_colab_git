# 模型验证
from ultralytics.models import YOLO
from ultralytics import settings
if __name__ == '__main__':
    # Update a setting
    settings.update({"datasets_dir": "D:\\study\\2025_gradproj\\Anti-UAV"})
    model = YOLO("./runs/train_2025-02-05_08-29-53/weights/best.pt")

    # Validate with a custom dataset
    metrics = model.val(data="./datasets/data.yaml")
    print(metrics.box.map)  # map50-95