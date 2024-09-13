#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class RobotController:
    def __init__(self):
        rospy.init_node('robot_controller', anonymous=True)
        self.velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.camera_subscriber = rospy.Subscriber('/camera/image_raw', Image, self.camera_callback)
        self.bridge = CvBridge()

    def move(self, linear_speed, angular_speed, duration):
        vel_msg = Twist()
        vel_msg.linear.x = linear_speed
        vel_msg.angular.z = angular_speed

        start_time = rospy.Time.now()
        while (rospy.Time.now() - start_time).to_sec() < duration:
            self.velocity_publisher.publish(vel_msg)
            rospy.sleep(0.1)

        # Stop the robot
        vel_msg.linear.x = 0
        vel_msg.angular.z = 0
        self.velocity_publisher.publish(vel_msg)

    def camera_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            cv2.imshow("Robot Camera", cv_image)
            cv2.waitKey(1)
        except Exception as e:
            print(e)

    def forward(self):
        self.move(0.2, 0, 2)  # Move forward at 0.2 m/s for 2 seconds

    def backward(self):
        self.move(-0.2, 0, 2)  # Move backward at 0.2 m/s for 2 seconds

    def rotate_left(self):
        self.move(0, 0.5, 2)  # Rotate left at 0.5 rad/s for 2 seconds

    def rotate_right(self):
        self.move(0, -0.5, 2)  # Rotate right at 0.5 rad/s for 2 seconds

if __name__ == '__main__':
    try:
        controller = RobotController()
        rospy.sleep(1)  # Wait for connections to be established

        print("Moving forward")
        controller.forward()
        rospy.sleep(1)

        print("Moving backward")
        controller.backward()
        rospy.sleep(1)

        print("Rotating left")
        controller.rotate_left()
        rospy.sleep(1)

        print("Rotating right")
        controller.rotate_right()

        # Keep the script running to continue receiving camera feed
        rospy.spin()

    except rospy.ROSInterruptException:
        pass