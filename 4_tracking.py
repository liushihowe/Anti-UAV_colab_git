import cv2
from ultralytics import YOLO

from collections import defaultdict 
# Load the YOLO11 model
model = YOLO("./runs/train_yoloi.yaml_2025-03-02_14-12-09/weights/best.pt")
# model = YOLO("./runs/train_yolon.yaml_2025-03-03_08-03-43/weights/best.pt")
video_path = "./datasets/测试/V_BIRD_005.mp4"
video_path = "D:\study\\2025_gradproj\Anti-UAV_colab_git\datasets\测试\V_DRONE_099.mp4"

results = model.track(source=video_path, conf=0.15, iou=0.5, show=True,save=True, persist=True, tracker="./models/bytetrack.yaml")

'''
cap = cv2.VideoCapture(video_path)

track_history = defaultdict(lambda: [])

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO11 tracking on the frame, persisting tracks between frames
        results = model.track(source=frame, imgsz=(640,256), batch=64, vid_stride=1, project="runs", name="track", 
                              save=False, persist=True, tracker="./models/bytetrack.yaml")
        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        # Display the annotated frame

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        # track_ids = results[0].boxes.id.int().cpu().tolist()

        cv2.imshow("YOLO11 Tracking", annotated_frame)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break
'''
