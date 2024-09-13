#!/usr/bin/env python3

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from std_msgs.msg import Float32
from cv_bridge import CvBridge
import collections

# Parameters for distance calculation (example focal length and known object width)
FOCAL_LENGTH = 615  # Focal length in pixels (depends on your camera)
KNOWN_WIDTH = 0.5  # Known width of the object in meters (adjust based on the object)

# Initialize ROS node
rospy.init_node('camera_raw_subscriber', anonymous=True)

# Create a CvBridge object
bridge = CvBridge()

# Publishers for distance and centroid
distance_pub = rospy.Publisher('/obstacle_distance', Float32, queue_size=10)
centroid_pub = rospy.Publisher('/obstacle_centroid', Point, queue_size=10)

# Global variables for processing frames
kernel_size = (5, 5)
threshold = 50
epsilon_factor = 0.02
min_contour_area = 1000  # Minimum contour area to be considered an obstacle
max_detection_distance = 5.0  # Maximum distance to consider an obstacle

# Global variables to hold the last known distance
last_known_distance = None
distance_history = collections.deque(maxlen=5)  # Store the last 5 distance values for smoothing

def smooth_distance(distance):
    distance_history.append(distance)
    return np.mean(distance_history)

def image_callback(msg):
    global last_known_distance
    
    # Convert ROS Image message to OpenCV format
    frame = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    # Convert to grayscale
    image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(image_gray, kernel_size, 0)
    
    # Convert to binary
    _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
    
    # Invert the binary image
    binary_inverted = cv2.bitwise_not(binary)
    
    # Find contours
    contours, _ = cv2.findContours(binary_inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a copy of the original image to draw contours on
    contour_image = frame.copy()

    if contours:
        # Process the largest contour (assumed to be the obstacle)
        contour = max(contours, key=cv2.contourArea)
        
        # Ignore small contours
        if cv2.contourArea(contour) < min_contour_area:
            return

        # Approximate the contour to a simpler shape
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Draw the approximated contour on the image
        cv2.drawContours(contour_image, [approx], -1, (0, 255, 0), 2)
        
        # Get the bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Compute centroid coordinates
        moments = cv2.moments(contour)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
        else:
            cx, cy = 0, 0
        
        # Draw the centroid
        cv2.circle(contour_image, (cx, cy), 5, (0, 0, 255), -1)  # Red circle for centroid
        
        # Estimate the distance
        distance = (KNOWN_WIDTH * FOCAL_LENGTH) / w
        distance = smooth_distance(distance)  # Apply smoothing
        last_known_distance = distance  # Update the last known distance
        
        if distance > max_detection_distance:
            return  # Ignore far away objects

        # Publish the centroid
        centroid_msg = Point(x=cx, y=cy, z=0)
        centroid_pub.publish(centroid_msg)

        # Publish the distance
        distance_pub.publish(distance)

        # Display distance and centroid information
        label = f"Distance: {distance:.2f} m, Centroid: ({cx}, {cy})"
        cv2.putText(contour_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    else:
        # If no contours are detected, publish the last known distance
        if last_known_distance is not None:
            distance_pub.publish(last_known_distance)
            rospy.loginfo(f"Using last known distance: {last_known_distance:.2f} meters")

    # Show the result
    cv2.imshow('Live Camera Feed with Distance and Centroid', contour_image)
    cv2.waitKey(1)

# Subscribe to the ROS camera topic
rospy.Subscriber('/camera/image_raw', Image, image_callback)

# Spin to keep the script running
rospy.spin()

# Cleanup
cv2.destroyAllWindows()
