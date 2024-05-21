import cv2
import numpy as np
import time

# ArUco dictionary
ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

def aruco_display(corners, ids, rejected, image):
    """
    Display ArUco markers on the image.
    """
    if len(corners) > 0:
        ids = ids.flatten()
        for (markerCorner, markerID) in zip(corners, ids):
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
            
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
            
            cv2.putText(image, str(markerID),(topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2)

    return image
def aruco_display_with_flower(frame, corners, ids, rejected):
    """
    Display ArUco markers on the frame along with a flower to the right of marker 1.
    """
    if len(corners) > 0:
        ids = ids.flatten()
        for i, markerID in enumerate(ids):
            # Draw ArUco marker
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            # If markerID is 1, draw a flower to the right
            if markerID == 1:
                draw_flower(frame, corners[i][0])

    return frame
def draw_flower(frame, radius, angle):
    """
    Draw a flower on the given frame.
    """
    # Create a black background
    background = np.zeros_like(frame)

    # Center coordinates
    center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2

    # Draw petals of the flower
    num_petals = 8
    for i in range(num_petals):
        start_angle = i * (360 / num_petals) + angle
        end_angle = start_angle + 45
        start_point = (int(center_x + radius * np.cos(np.radians(start_angle))),
                       int(center_y + radius * np.sin(np.radians(start_angle))))
        end_point = (int(center_x + radius * np.cos(np.radians(end_angle))),
                     int(center_y + radius * np.sin(np.radians(end_angle))))
        cv2.line(background, start_point, end_point, (255, 255, 255), 2)
        
    # Draw center of the flower smaller circle
    cv2.circle(background, (center_x, center_y), radius // 8, (255, 255, 255), -1)
    
    return background
def draw_flower(frame, marker_corner):
    """
    Draw a flower to the right of the given ArUco marker corner.
    """
    # Define flower parameters
    radius = 50
    center_x, center_y = int(marker_corner[0][0]), int(marker_corner[0][1])
    flower_center_x = center_x + radius * 2
    flower_center_y = center_y

    # Draw flower petals
    num_petals = 8
    for i in range(num_petals):
        start_angle = i * (360 / num_petals)
        end_angle = start_angle + 45
        start_point = (int(flower_center_x + radius * np.cos(np.radians(start_angle))),
                       int(flower_center_y + radius * np.sin(np.radians(start_angle))))
        end_point = (int(flower_center_x + radius * np.cos(np.radians(end_angle))),
                     int(flower_center_y + radius * np.sin(np.radians(end_angle))))
        cv2.line(frame, start_point, end_point, (255, 255, 255), 2)

    # Draw flower center
    cv2.circle(frame, (flower_center_x, flower_center_y), radius // 8, (255, 255, 255), -1)
def main():
    # Set up video writers
    aruco_type = "DICT_5X5_100"

    arucoParams =  cv2.aruco.DetectorParameters()
    arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_type])

    # Set up video capture
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Create a window for the animation
    cv2.namedWindow('Animation', cv2.WINDOW_NORMAL)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        ret, img = cap.read()

        h, w, _ = img.shape

        width = 1000
        height = int(width*(h/w))
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
        
        # Detect ArUco markers
        corners, ids, rejected = cv2.aruco.detectMarkers(img, arucoDict, parameters=arucoParams)
        
        detected_markers = aruco_display(corners, ids, rejected, img)

    
        
        # Display the detected markers
        cv2.imshow('ArUco Detection', detected_markers)

        # Draw flower animation
        animated_frame = draw_flower(frame, radius=100, angle=(time.time() * 100) % 360)

        # Display the animated frame
        cv2.imshow('Animation', animated_frame)

        # Check for quit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and video writers
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
