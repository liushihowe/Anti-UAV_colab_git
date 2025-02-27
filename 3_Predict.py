from ultralytics.models import YOLO
import cv2 

 
if __name__ == '__main__':
    # model = YOLO(model='./test.pt')
    # results = model('./datasets/images/train/0011.jpg')
    # # Process results list
    # for result in results:
    #     boxes = result.boxes
    #     result.show() # display to screen
    #     result.save(filename="./runs/detect/result.jpg") # save to disk

    # Load a pretrained YOLO11n model
    model = YOLO("./runs/train_2025-02-05_08-29-53/weights/best.pt")


    # Run inference on the source
    

    # Read an image using OpenCV
    path = "./datasets/测试/94.bmp"
    source = cv2.imread(path)
    results = model.predict(source=source, save=False, imgsz=320, conf=0.5)  # list of Results objects
    results[0].show()  # display to screen

    # Open the video file
    video_path = "./datasets/测试/IR_BIRD_036.mp4"
    cap = cv2.VideoCapture(video_path)

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLO inference on the frame
            results = model(frame)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("YOLO Inference", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()