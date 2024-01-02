import cv2
import numpy as np

# Load the image of the beacon you want to track
beacon_image_path = 'D:\\github\\ComputerVision-OpenCV\\myscript\\beacon_drawn.jpg'
beacon_template = cv2.imread(beacon_image_path, 0)  # Load in grayscale

if beacon_template is None:
    raise ValueError(f"The image at {beacon_image_path} could not be loaded. Check the file path and integrity.")

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Find the keypoints and descriptors with SIFT in the template
keypoints_template, descriptors_template = sift.detectAndCompute(beacon_template, None)
if descriptors_template is not None:
    descriptors_template = np.float32(descriptors_template)

# FLANN parameters and matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Set up the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find the keypoints and descriptors with SIFT in the video frame
    keypoints_frame, descriptors_frame = sift.detectAndCompute(gray_frame, None)
    if descriptors_frame is not None:
        descriptors_frame = np.float32(descriptors_frame)

        # Match descriptors if both sets are available
        if descriptors_template is not None and descriptors_frame is not None:
            matches = flann.knnMatch(descriptors_template, descriptors_frame, k=2)

            # Store all good matches as per Lowe's ratio test
            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)

            if len(good_matches) > 10:  # Enough matches are found
                src_pts = np.float32([keypoints_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([keypoints_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                # Find homography
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                # Check if the homography matrix is valid
                if M is not None and M.shape == (3, 3):
                    matchesMask = mask.ravel().tolist()

                    # Draw a polygon around the detected region
                    h, w = beacon_template.shape
                    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                    dst = cv2.perspectiveTransform(pts, M)
                    frame = cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 255), 3, cv2.LINE_AA)
                else:
                    matchesMask = None

    # Display the resulting frame
    cv2.imshow('Object Tracking', frame)

    # Display the confidence map
    if descriptors_template is not None and descriptors_frame is not None:
        # The confidence map is actually the 'res' matrix from the matchTemplate function
        # Normalize the result to [0, 255] for display
        confidence_map = cv2.normalize(matches, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        cv2.imshow('Confidence Map', confidence_map)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
