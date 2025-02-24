# from ultralytics.models import YOLO
from ultralytics.nn.tasks import DetectionModel
# # class DetectionModel(BaseModel):


# # 模型网络结构配置文件路径
# yaml_path = './yolo11.yaml'
# # model = YOLO('./yolo11i.yaml')
# # model = YOLO('../runs/train_2025-02-05_08-29-53/weights/best.pt')
# # # print(model.info())
# # print(model.info(detailed=True))

# # # 改进的模型结构路径
# # # yaml_path = 'ultralytics/cfg/models/v8/yolov8n-CBAM.yaml'  
# # # 传入模型网络结构配置文件cfg, nc为模型检测类别数

# # model = YOLO('./yolo11i.yaml')
# DetectionModel(cfg=yaml_path, nc=8)

# 模型网络结构配置文件路径
yaml_path = './yolo11.yaml' 
# 传入模型网络结构配置文件cfg, nc为模型检测类别数
DetectionModel(cfg=yaml_path,nc=8)


yaml_path = './yolo11_n.yaml'
DetectionModel(cfg=yaml_path,nc=8)
