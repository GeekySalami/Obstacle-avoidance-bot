#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist

def turn_robot():
    # Initialize the ROS node
    rospy.init_node('turn_robot', anonymous=True)
    
    # Create a publisher for the /cmd_vel topic
    cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    
    # Define the Twist message for turning
    twist = Twist()
    
    # Set the turning rate (radians per second)
    turning_rate = 0.5  # Adjust this value for faster/slower turns

    # Set the duration of the turn (seconds)
    turn_duration = 5  # Adjust this value for how long to turn

    # Calculate the end time
    end_time = rospy.Time.now() + rospy.Duration(turn_duration)

    # Loop until the turn duration is reached
    while rospy.Time.now() < end_time:
        # Set angular velocity to turn left (positive value for counter-clockwise)
        twist.angular.z = turning_rate
        # Set linear velocity to 0 (no forward/backward movement)
        twist.linear.x = 0
        
        # Publish the Twist message
        cmd_vel_pub.publish(twist)
        
        # Sleep for a short duration to maintain the loop rate
        rospy.sleep(0.1)

    # Stop the robot after turning
    twist.angular.z = 0
    cmd_vel_pub.publish(twist)

if __name__ == '__main__':
    try:
        turn_robot()
    except rospy.ROSInterruptException:
        pass
