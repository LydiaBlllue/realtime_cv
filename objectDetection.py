import cv2
import torch
from yolov5 import YOLOv5  # This is assuming you have installed the yolov5 package

# Load the YOLOv5 model
model_path = "yolov5s.pt"  # This can be yolov5m, yolov5l, or yolov5x depending on your need for speed vs accuracy
model = YOLOv5(model_path, device="cuda" if torch.cuda.is_available() else "cpu")

# Initialize video capture
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection
    # The YOLOv5 package expects images in BGR format, which OpenCV uses by default
    results = model.predict(frame)
    
    # results.xyxy[0] contains the bounding box coordinates in the format [x1, y1, x2, y2, confidence, class]
    # Loop through detections and draw them on the frame
    for detection in results.xyxy[0]:
        x1, y1, x2, y2, conf, class_ = map(int, detection.tolist())
        label = model.names[class_]  # Get the name of the label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
