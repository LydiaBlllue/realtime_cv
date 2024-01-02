import cv2
import numpy as np

# Load the image of the beacon you want to track
beacon_image_path = 'D:\\github\\ComputerVision-OpenCV\\myscript\\beacon_drawn.jpg'
beacon_template = cv2.imread(beacon_image_path, 0)  # Load in grayscale

if beacon_template is None:
    raise ValueError(f"The image at {beacon_image_path} could not be loaded. Check the file path and integrity.")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(gray_frame, beacon_template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    if max_val >= 0.7:  # Adjust this threshold as necessary
        w, h = beacon_template.shape[::-1]
        cv2.rectangle(frame, max_loc, (max_loc[0] + w, max_loc[1] + h), (0, 255, 255), 2)

    # Display the resulting frame with detections
    cv2.imshow('Object Tracking', frame)

    # If you want to show the match confidence map, normalize the result first
    # This step converts the match result to a visual format that's easier to see
    res_display = cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    cv2.imshow('Template Matching Confidence Map', res_display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
