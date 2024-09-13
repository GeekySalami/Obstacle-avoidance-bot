#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float64
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import sys
import select
import termios
import tty

class PriusController:
    def __init__(self):
        rospy.init_node('prius_controller')

        self.throttle_pub = rospy.Publisher('/prius/throttle_cmd', Float64, queue_size=1)
        self.brake_pub = rospy.Publisher('/prius/brake_cmd', Float64, queue_size=1)
        self.steer_pub = rospy.Publisher('/prius/steering_cmd', Float64, queue_size=1)
        self.gear_pub = rospy.Publisher('/prius/gear_cmd', Float64, queue_size=1)

        rospy.Subscriber("/prius/front_camera/image_raw", Image, self.image_callback)
        rospy.Subscriber("/prius/joint_states", JointState, self.joint_states_callback)

        self.bridge = CvBridge()
        self.cv_image = None
        self.throttle = 0
        self.brake = 0
        self.steer = 0
        self.gear = 0
        self.speed = 0

        self.settings = termios.tcgetattr(sys.stdin)

    def image_callback(self, msg):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)

    def joint_states_callback(self, msg):
        if 'wheel_rear_left' in msg.name:
            idx = msg.name.index('wheel_rear_left')
            self.speed = msg.velocity[idx] * 0.1  # Approximate conversion to m/s

    def get_key(self):
        tty.setraw(sys.stdin.fileno())
        select.select([sys.stdin], [], [], 0)
        key = sys.stdin.read(1)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key

    def update_commands(self, key):
        if key in ['a', 's', 'd']:
            self.gear = {'a': 0, 's': 1, 'd': -1}[key]
            self.gear_pub.publish(Float64(self.gear))
        elif key in ['q', 'z']:
            self.throttle += 0.1 if key == 'q' else -0.1
            self.throttle = max(min(1, self.throttle), 0)
            self.throttle_pub.publish(Float64(self.throttle))
        elif key in ['w', 'x']:
            self.brake += 0.1 if key == 'w' else -0.1
            self.brake = max(min(1, self.brake), 0)
            self.brake_pub.publish(Float64(self.brake))
        elif key in ['e', 'c']:
            self.steer += 0.1 if key == 'e' else -0.1
            self.steer = max(min(0.7, self.steer), -0.7)
            self.steer_pub.publish(Float64(self.steer))

    def display_info(self):
        if self.cv_image is not None:
            cv2.imshow("Prius Front Camera", self.cv_image)
            cv2.waitKey(3)

        info = f"Speed: {self.speed:.2f} m/s | Throttle: {self.throttle:.2f} | Brake: {self.brake:.2f} | Steer: {self.steer:.2f} | Gear: {self.gear}"
        print(info, end='\r')

    def run(self):
        print("""
Control Your Prius!
---------------------------
Moving around:
   q    w    e
   a    s    d
   z    x    c

q/z : increase/decrease throttle
w/x : increase/decrease brake
e/c : increase/decrease steering
a : neutral gear
s : drive gear
d : reverse gear

CTRL-C to quit
        """)

        try:
            while not rospy.is_shutdown():
                key = self.get_key()
                if key == '\x03':  # CTRL-C
                    break
                self.update_commands(key)
                self.display_info()
        except Exception as e:
            print(e)
        finally:
            self.cleanup()

    def cleanup(self):
        self.throttle_pub.publish(Float64(0))
        self.brake_pub.publish(Float64(0))
        self.steer_pub.publish(Float64(0))
        self.gear_pub.publish(Float64(0))
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    controller = PriusController()
    controller.run()
