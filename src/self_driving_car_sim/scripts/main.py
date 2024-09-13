#!/usr/bin/env python3

import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from objectdetection import ObjectTracker

def image_callback(image_msg):
    try:
        # Convert ROS Image message to OpenCV image
        cv_image = bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
    except CvBridgeError as e:
        print(e)
        return

    # Detect objects and track them
    results = tracker.detect_and_track(cv_image)

    # Display detection results
    for result in results:
        startX, startY, endX, endY = result["bounding_box"]
        label = f"{result['class']} ({result['confidence']:.2f}), Dist: {result['distance']:.2f}m"
        if result['speed'] is not None:
            label += f", Speed: {result['speed']:.2f}m/s"

        # Draw bounding box and label on the image
        cv2.rectangle(cv_image, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.putText(cv_image, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the output image
    cv2.imshow("Frame", cv_image)
    cv2.waitKey(1)

if __name__ == "__main__":
    rospy.init_node('object_detection_node', anonymous=True)

    # Known real-world dimensions for cubes and cuboids (in meters)
    known_dimensions = {
        "person": 0.5,   # Example: average shoulder width of a person
        "car": 1.8,      # Example: width of a car
        "bottle": 0.07,  # Example: width of a bottle
        "box": 0.5,      # Example: width of a cuboidal box
    }

    # Initialize ObjectTracker
    tracker = ObjectTracker(
        '/home/sharvil-palvekar/catkin_ws/src/self_driving_car_sim/scripts/MobileNet-SSD/deploy.prototxt',
        '/home/sharvil-palvekar/catkin_ws/src/self_driving_car_sim/scripts/MobileNet-SSD/mobilenet_iter_73000.caffemodel',
        known_dimensions
    )

    # Initialize CvBridge
    bridge = CvBridge()

    # Subscribe to the ROS image topic
    image_topic = "/camera/image_raw"  # Change this to your actual topic
    rospy.Subscriber(image_topic, Image, image_callback)

    # Keep the node running
    rospy.spin()

    # Cleanup
    cv2.destroyAllWindows()
