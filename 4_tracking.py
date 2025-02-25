import cv2
from ultralytics import YOLO

from collections import defaultdict 
# Load the YOLO11 model
model = YOLO("./runs/train_2025-02-05_08-29-53/weights/best.pt")
video_path = "datasets/测试/IR_BIRD_036.mp4"
cap = cv2.VideoCapture(video_path)

track_history = defaultdict(lambda: [])

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO11 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, tracker="./models/botsort.yaml")
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

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()


