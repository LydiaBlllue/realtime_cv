import cv2
import numpy as np

# Set the dimensions of the image (width, height)
width, height = 200, 200

# Create a white background image
image = np.ones((height, width, 3), dtype=np.uint8) * 255

# Define the thickness
thickness = 2
# Calculate the center and radius
center_coordinates = (width // 2, height // 2)
radius = width // 2

# Draw a black circle to form the base
cv2.circle(image, center_coordinates, radius, (0, 0, 0), thickness)

# Draw a black line from the center to the top, bottom, left and right edges
cv2.line(image, center_coordinates, (center_coordinates[0], 0), (0, 0, 0), thickness)
cv2.line(image, center_coordinates, (center_coordinates[0], height), (0, 0, 0), thickness)
cv2.line(image, center_coordinates, (0, center_coordinates[1]), (0, 0, 0), thickness)
cv2.line(image, center_coordinates, (width, center_coordinates[1]), (0, 0, 0), thickness)

# Fill the second and third quadrant with black on the circle base
# Define the points for the second and third quadrants
pts_second_quad = np.array([[center_coordinates[0], center_coordinates[1]], 
                            [width, center_coordinates[1]], 
                            [width, 0], 
                            [center_coordinates[0], 0]], np.int32)

pts_third_quad = np.array([[center_coordinates[0], center_coordinates[1]], 
                           [0, center_coordinates[1]], 
                           [0, height], 
                           [center_coordinates[0], height]], np.int32)

# Draw the filled quadrants
cv2.fillPoly(image, [pts_second_quad], (0, 0, 0))
cv2.fillPoly(image, [pts_third_quad], (0, 0, 0))

# Save the image
beacon_image_path = 'beacon_drawn.jpg'
cv2.imwrite(beacon_image_path, image)

# Display the image
cv2.imshow('Beacon Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
