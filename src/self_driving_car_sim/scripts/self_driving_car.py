#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
from tf.transformations import euler_from_quaternion
import numpy as np

class SelfDrivingCar:
    def __init__(self):
        rospy.init_node('self_driving_car', anonymous=True)

        # Subscribers for obstacle distance, centroid, and odometry
        self.distance_sub = rospy.Subscriber('/obstacle_distance', Float32, self.distance_callback)
        self.centroid_sub = rospy.Subscriber('/obstacle_centroid', Point, self.centroid_callback)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)  # Subscribe to odom

        # Publisher for controlling the car's movement
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.twist = Twist()

        # Initialize parameters for navigation
        self.goal = Point(-12.0, 0.0, 0)  # Example goal (x=-12.0, y=0.0)
        self.current_position = Point()  # Current position (starting at origin)
        self.current_heading = 0.0  # Current heading (yaw)
        self.obstacle_distance = None  # Distance to the nearest obstacle
        self.obstacle_centroid = None  # Centroid of the detected obstacle

        # Obstacle avoidance parameters
        self.min_safe_distance = 0.76  # Minimum safe distance from obstacles in meters
        self.avoidance_turn_rate = 0.9  # Rate to turn when avoiding obstacles
        self.max_linear_speed = 0.7  # Maximum linear speed in m/s

    def distance_callback(self, msg):
        self.obstacle_distance = msg.data

    def centroid_callback(self, msg):
        self.obstacle_centroid = msg

    def odom_callback(self, msg):
        """
        Callback to update the current position and heading of the robot from odometry data.
        """
        # Update current position
        self.current_position = msg.pose.pose.position
        
        # Get the robot's orientation (quaternion)
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]

        # Convert quaternion to Euler angles to get the yaw (heading)
        _, _, self.current_heading = euler_from_quaternion(orientation_list)

    def navigate_to_goal(self):
        """
        Move towards the goal while avoiding obstacles.
        """
        if self.obstacle_distance and self.obstacle_distance < self.min_safe_distance:
            # If an obstacle is too close, perform obstacle avoidance
            rospy.loginfo("Obstacle detected! Avoiding...")
            self.avoid_obstacle()
        else:
            # If no obstacles are detected, move towards the goal
            self.move_towards_goal()

    def move_towards_goal(self):
        """
        Moves the car towards the goal using proportional control.
        """
        # Calculate the direction to the goal
        goal_direction = np.arctan2(self.goal.y - self.current_position.y, self.goal.x - self.current_position.x)
        
        # Calculate the distance to the goal
        goal_distance = np.sqrt((self.goal.x - self.current_position.x)**2 + (self.goal.y - self.current_position.y)**2)

        if goal_distance > 0.1:
            # Proportional control for linear velocity
            self.twist.linear.x = min(self.max_linear_speed, 0.8 * goal_distance)

            # Calculate the heading error
            heading_error = goal_direction - self.current_heading
            
            # Normalize the heading error to the range [-pi, pi]
            heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

            # Proportional control for angular velocity
            self.twist.angular.z = 2.0 * heading_error  # Adjust the 2.0 gain as needed

            # Limit angular velocity to prevent fast spinning
            self.twist.angular.z = np.clip(self.twist.angular.z, -1.0, 1.0)  # Limit between -1.0 and 1.0 radians/second
        else:
            # Stop if we are close enough to the goal
            rospy.loginfo("Reached the goal!")
            self.twist.linear.x = 0
            self.twist.angular.z = 0

        # Publish the twist message
        self.cmd_vel_pub.publish(self.twist)


    def avoid_obstacle(self):
        """
        Avoids obstacles based on the centroid of the detected obstacle.
        """
        if self.obstacle_centroid is not None:
            rospy.loginfo(f"Obstacle Centroid: x={self.obstacle_centroid.x}, y={self.obstacle_centroid.y}")
        
            if self.obstacle_centroid.x < 200:
                # If the obstacle is to the left, turn right
                self.twist.angular.z = -1.2 * self.avoidance_turn_rate  # Increase turn rate
                rospy.loginfo("Turning right to avoid obstacle")
            else:
                # If the obstacle is to the right, turn left
                self.twist.angular.z = 1.2 * self.avoidance_turn_rate  # Increase turn rate
                rospy.loginfo("Turning left to avoid obstacle")

            # Slow down when avoiding obstacles
            self.twist.linear.x = 0.1  # Slow down more
            rospy.loginfo(f"Obstacle detected! Slowing down to {self.twist.linear.x} m/s")
        else:
            # Default behavior when no centroid is available (go straight)
            self.twist.angular.z = 0.0
            self.twist.linear.x = 0.3  # Default speed
            rospy.loginfo("No obstacle detected, moving straight")

        # Publish the twist message
        self.cmd_vel_pub.publish(self.twist)

    def run(self):
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            self.navigate_to_goal()
            rate.sleep()

if __name__ == '__main__':
    try:
        car = SelfDrivingCar()
        car.run()
    except rospy.ROSInterruptException:
        pass
