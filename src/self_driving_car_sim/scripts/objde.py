#!/usr/bin/env python3

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from std_msgs.msg import Float32
from cv_bridge import CvBridge
import collections

FOCAL_LENGTH = 615
KNOWN_WIDTH = 0.5

rospy.init_node('camera_raw_subscriber', anonymous=True)

bridge = CvBridge()

distance_pub = rospy.Publisher('/obstacle_distance', Float32, queue_size=10)
centroid_pub = rospy.Publisher('/obstacle_centroid', Point, queue_size=10)

kernel_size = (5, 5)
threshold = 50
epsilon_factor = 0.02
min_contour_area = 1000
max_detection_distance = 5.0

distance_history = collections.deque(maxlen=5)

def smooth_distance(distance):
    distance_history.append(distance)
    return np.mean(distance_history)

def image_callback(msg):
    frame = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    blurred = cv2.GaussianBlur(image_gray, kernel_size, 0)
    
    _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
    
    binary_inverted = cv2.bitwise_not(binary)
    
    contours, _ = cv2.findContours(binary_inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    contour_image = frame.copy()

    if contours:
        contour = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(contour) < min_contour_area:
            return

        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        cv2.drawContours(contour_image, [approx], -1, (0, 255, 0), 2)
        
        x, y, w, h = cv2.boundingRect(contour)
        
        moments = cv2.moments(contour)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
        else:
            cx, cy = 0, 0
        
        cv2.circle(contour_image, (cx, cy), 5, (0, 0, 255), -1)
        
        distance = (KNOWN_WIDTH * FOCAL_LENGTH) / w
        distance = smooth_distance(distance)
        
        if distance > max_detection_distance:
            return

        centroid_msg = Point(x=cx, y=cy, z=0)
        centroid_pub.publish(centroid_msg)

        distance_pub.publish(distance)

        label = f"Distance: {distance:.2f} m, Centroid: ({cx}, {cy})"
        cv2.putText(contour_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    img = contour_image
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow('Live Camera Feed with Distance and Centroid', img)
    cv2.waitKey(1)

rospy.Subscriber('/camera/image_raw', Image, image_callback)

rospy.spin()

cv2.destroyAllWindows()
