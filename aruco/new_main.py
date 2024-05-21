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
    # Set up video capture
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame for better performance
        frame = cv2.resize(frame, (640, 480))

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Initialize ArUco detector parameters
        parameters = cv2.aruco.DetectorParameters()
        # Load ArUco dictionary
        aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT["DICT_5X5_100"])

        # Detect ArUco markers
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        # Display ArUco markers with flower
        frame_with_markers = aruco_display_with_flower(frame, corners, ids, rejected)

        # Show the frame
        cv2.imshow('ArUco Marker with Flower', frame_with_markers)

        # Check for quit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
