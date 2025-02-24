# pip install ultralytics

from ultralytics.models import YOLO
import torch
# 自动选择设备，GPU 优先，如果没有 GPU，则使用 CPU
device = torch.device(0 if torch.cuda.is_available() else 'cpu')

import torch
torch.cuda.empty_cache()

# 获取当前时间并格式化
from datetime import datetime
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
base_path = "train"
path = f"{base_path}_{current_time}"
print(f"动态路径：{path}")

if __name__ == '__main__':

    model = YOLO("./models/yolo11_n.yaml") 
    # Train the model
    results = model.train(data='datasets\data.yaml', epochs=50, batch=0.7, 
                          imgsz=320, save=True, cache='disk', device=device, workers = 0,
                          project='runs', name=path, exist_ok=False, pretrained=True, single_cls=False, 
                          resume=False, val=False)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base = "model"
    model_path = f"{base}_{current_time}"
    model.save(f"./models/{model_path}")