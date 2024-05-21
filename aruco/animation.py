import cv2
import numpy as np
import time

def draw_flower(frame, radius, angle):
    # Create a black background
    background = np.zeros_like(frame)

    # Center coordinates
    center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2

    # Draw petals of the flower
    # flower is  white

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

def animation():
    # Define parameters
    radius = 100
 
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('flower_animation.mp4', fourcc, 30.0, (640, 480))\
    
    #while true, unitil key q is pressed
    while True:
        # Create a black background
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Draw flower
        angle = (time.time() * 100) % 360
        animated_frame = draw_flower(frame, radius, angle)
        
        # Write frame to the output video
        out.write(animated_frame)
        
        # Display the animated frame
        cv2.imshow('Animation', animated_frame)
        
        # Check for quit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video writer
    out.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    animation()