#!/usr/bin/env python3

import cv2
import numpy as np
import rospy
import yaml

from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CameraInfo, CompressedImage
from std_msgs.msg import Float32, String, Bool
from turbojpeg import TurboJPEG

from duckietown_msgs.msg import WheelsCmdStamped, Twist2DStamped
import tf.transformations as tft
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from duckietown_msgs.srv import ChangePattern


# Define the HSV color range for road and stop mask
ROAD_MASK = [(20, 60, 0), (50, 255, 255)]
STOP_MASK = [(0, 120, 120), (15, 255, 255)]
NUM_MASK = [(90, 85, 46), (175, 255, 255)]

# Turn pattern 
TURNS = ['S', 'S', 'L', 'R', 'S', 'R', 'L'] # starting at apriltag 3
TURN_VALUES = {'S': 0, 'L': np.pi/2, 'R': -np.pi/2}
TURNS_RADIUS = [0, 0, 0.3, 0.12, 0, 0.12, 0.3]
STOP_RED = False
STOP_BLUE = False

# Set debugging mode (change to True to enable)
DEBUG = False

class LaneFollowNode(DTROS):
    def __init__(self, node_name):
        super(LaneFollowNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

        # Save node name and vehicle name
        self.node_name = node_name
        self.veh = rospy.get_param("~veh")

        # Publishers & Subscribers
        self.pub = rospy.Publisher("/" + self.veh + "/output/image/mask/compressed",
                                   CompressedImage,
                                   queue_size=1)
        self.sub = rospy.Subscriber("/" + self.veh + "/camera_node/image/compressed",
                                    CompressedImage,
                                    self.callback,
                                    queue_size=1,
                                    buff_size="20MB")
                                    
        self.sub_done = rospy.Subscriber("/" + self.veh + "/done",
                                    Bool,
                                    self.cb_done,
                                    queue_size=1)   

        # Initialize distance subscriber and velocity publisher
        #self.sub_ml = rospy.Subscriber("/" + self.veh + "/augmented_reality_node/position", Point, self.cb_april, queue_size=1)
        self.vel_pub = rospy.Publisher("/" + self.veh + "/car_cmd_switch_node/cmd",
                                       Twist2DStamped,
                                       queue_size=1)
        
        # Initialize TurboJPEG decoder
        self.jpeg = TurboJPEG()
        

        # PID Variables
        self.proportional = None
        self.proportional_stopline = None
        self.offset = 200  # 220

        self.velocity = 0.3
        self.twist = Twist2DStamped(v=self.velocity, omega=0)

        self.P = 0.035
        self.D = -0.0025
        self.last_error = 0
        self.last_time = rospy.get_time()

        self.kp_turn = 1

        # number area 
        self.last_num_area = 0

        # Robot Pose Variables
        self.displacement = 0
        self.orientation = 0
        
        self.subscriber = rospy.Subscriber(f'/{self.veh}/deadreckoning_node/odom', 
                                           Odometry, 
                                           self.odom_callback)
                                           
        self.subscriber = rospy.Subscriber(f'/{self.veh}/deadreckoning_node/odom', 
                                           Odometry, 
                                           self.odom_callback)

        # Initialize LED pattern change service
        led_service = f'/{self.veh}/led_controller_node/led_pattern'
        rospy.wait_for_service(led_service)
        self.led_pattern = rospy.ServiceProxy(led_service, ChangePattern)

        # Initialize shutdown hook
        rospy.on_shutdown(self.hook)


    def color_mask(self, img, mask, crop_width, pid=False, stopping=False, number=False):
        global STOP_RED, STOP_BLUE
        # Convert the cropped image to HSV color space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Apply the color mask to the HSV image
        mask = cv2.inRange(hsv, mask[0], mask[1])
        crop = cv2.bitwise_and(img, img, mask=mask)

        if number:
            # Define the structuring element for dilation
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

            # Erode the image
            eroded_image = cv2.erode(crop, kernel, iterations=1)

            # Dilate the eroded image
            crop = cv2.dilate(eroded_image, kernel, iterations=2)

        # Find contours in the masked image
        contours_road, hierarchy = cv2.findContours(mask,
                                            cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_NONE)

        # Find the largest contour
        max_area = 20
        max_idx = -1
        for i in range(len(contours_road)):
            area = cv2.contourArea(contours_road[i])
            if area > max_area:
                max_idx = i
                max_area = area

        # If a contour is found
        if max_idx != -1:
            # Calculate the centroid of the contour
            M = cv2.moments(contours_road[max_idx])
            try:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                # If using PID control, updat e the proportional term
                if pid:
                    # check x cord (inside lane problem)
                    self.proportional = cx - int(crop_width / 2) + self.offset

                # If checking for stopping condition or below the threshold, set STOP_RED
                elif stopping:
                    # print('stoping cond cy: ', cy, cx)
                    if cx > 50:
                        self.proportional_stopline = (cy/168)*0.12
                    else: 
                        self.proportional_stopline = None

                    if cy >= 20:
                        STOP_BLUE = False

                    if cy >= 170 and cx in range(340, 645):
                        STOP_RED = True

                # If checking for number condition or above the threshold, set STOP_BLUE
                elif number and max_area < 2500:
                    # print('max_area', max_area)

                    if max_area >= 2100:
                        # print('cy: ', cy, 'cx: ', cx)
                        STOP_BLUE = True

                else:
                    STOP_RED = STOP_BLUE = False

                # Draw the contour and centroid on the image (for debugging)
                if DEBUG:
                    cv2.drawContours(crop, contours_road, max_idx, (0, 255, 0), 3)
                    cv2.circle(crop, (cx, cy), 7, (0, 0, 255), -1)
            except:
                pass

        # If no contour is found, reset the flags
        else:
            if pid:
                self.proportional = None

            elif stopping:
                self.proportional_stopline = None
                STOP_RED = False

            elif number:
                STOP_BLUE = False


    def callback(self, msg):
        # Decode the JPEG image from the message
        img = self.jpeg.decode(msg.data)
        h, w, _ = img.shape
        # Crop the image to focus on the road
        crop_road = img[300:-1, :, :]
        crop_sign = img[:, 200:w-100, :]

        crop_width = crop_road.shape[1]

        # Process the image for PID control using the ROAD_MASK
        self.color_mask(crop_road, ROAD_MASK, crop_width, pid=True)
        # Process the image for number stopping condition using the NUM_MASK
        self.color_mask(crop_sign, NUM_MASK, crop_width, number=True)
        # Process the image for stopping condition using the STOP_MASK
        self.color_mask(crop_road, STOP_MASK, crop_width, stopping=True)


    def odom_callback(self, data):
        orientation_quaternion = data.pose.pose.orientation
        _, _, yaw = tft.euler_from_quaternion([
            orientation_quaternion.x,
            orientation_quaternion.y,
            orientation_quaternion.z,
            orientation_quaternion.w
        ])
        self.orientation = yaw

        # Get position data
        position = data.pose.pose.position
        x, y, z = position.x, position.y, position.z

        # Calculate displacement from the origin
        displacement = np.sqrt(x**2 + y**2 + z**2)
        self.displacement = displacement


    def turn(self, r, target_angle):

        # Calculate the target orientation 
        target_orientation = self.orientation + target_angle
        orientation_error_raw = target_orientation - self.orientation
        orientation_error = np.sign(orientation_error_raw) * (abs(orientation_error_raw) % (2*np.pi))

        # Create a rospy.Rate object to maintain the loop rate at 8 Hz
        rate = rospy.Rate(8)

        while abs(orientation_error) > 0.2:
            print('orientation_error: ', np.rad2deg(orientation_error))

            orientation_error_raw = target_orientation - self.orientation
            orientation_error = np.sign(orientation_error_raw) * (abs(orientation_error_raw) % (2*np.pi))

            # Calculate the angular speed using a simple controller
            # angular_speed = 0.296 * np.log(np.sign(orientation_error) * orientation_error + 0.06) +0.42
            if r == 0.3:
                angular_speed = 3.2 * np.sign(orientation_error)
                linear_speed = 0.3
            else:
                angular_speed = 4 * np.sign(orientation_error)
                linear_speed = 0.32

            # print('linear_speed', linear_speed)

            # Set the linear and angular speeds in the Twist message
            self.twist.v = linear_speed
            self.twist.omega = angular_speed

            self.vel_pub.publish(self.twist)

            rate.sleep()


    def drive(self):
        if self.proportional is None:
            self.twist.omega = 0
            self.last_error = 0

        else:
            # P Term
            P = -self.proportional * self.P

            # D Term
            d_error = (self.proportional - self.last_error) / (rospy.get_time() - self.last_time)
            self.last_error = self.proportional
            self.last_time = rospy.get_time()
            D = d_error * self.D
            self.twist.omega = P + D

            if DEBUG:
                print(self.proportional, P, D, self.twist.omega, self.twist.v)

        if self.proportional_stopline is None:
            self.twist.v = self.velocity

        else:
            self.twist.v = self.velocity - self.proportional_stopline
        
        self.vel_pub.publish(self.twist)


    def traverse_town(self):
        global STOP_RED, STOP_BLUE
        rate = rospy.Rate(8)  # 8hz

        turn = 0

        while not rospy.is_shutdown():
            turn = turn % len(TURNS)
            
            # Continue driving until a stop sign (red) or a number sign (blue) is detected
            while not STOP_RED and not STOP_BLUE:
                self.drive()
                rate.sleep()

            # Stop the Duckiebot once a sign is detected
            before_stop_red, before_stop_blue = STOP_RED, STOP_BLUE
            self.move_robust(speed=0 ,seconds=2)
            STOP_RED, STOP_BLUE = before_stop_red, before_stop_blue

            print('STOP_RED', STOP_RED)
            print('STOP_BLUE', STOP_BLUE)
            print(TURNS[turn])

            # If a stop line is detected
            if STOP_RED:
                turning_angle = TURN_VALUES[TURNS[turn]]
                if turning_angle == 0:
                    self.move_robust(speed=0.3 ,seconds=2.5)
                    turn += 1
                    STOP_RED = False

                else:
                    self.turn(TURNS_RADIUS[turn], turning_angle)
                    turn += 1
                    STOP_RED = False

            # If a number sign is detected
            elif STOP_BLUE:
                # sleep? wait for number? Sergy? 
                self.move_robust(speed=0 ,seconds=2)
                self.lane_follow_n_sec(2)
                STOP_BLUE = False
                

    def move_robust(self, speed, seconds):
        rate = rospy.Rate(10)
        #print('speed - robust: ', speed)

        self.twist.v = speed
        self.twist.omega = 0

        # Publish the twist message multiple times to ensure the robot stops
        for i in range(int(10*seconds)):
            self.vel_pub.publish(self.twist)
            rate.sleep()


    def lane_follow_n_sec(self, seconds):
        rate = rospy.Rate(8)  # 8hz

        for i in range(int(8*seconds)):
            self.drive()
            rate.sleep()

    def cb_done(self, done):
        if done.data:
            self.hook()

    def hook(self):
        print("SHUTTING DOWN")
        self.move_robust(0, 2)


    def change_led_lights(self, dir: str):
        '''
        Sends msg to service server
        ignore
        Colors:
            "off": [0,0,0],
            "white": [1,1,1],
            "green": [0,1,0],
            "red": [1,0,0],
            "blue": [0,0,1],
            "yellow": [1,0.8,0],
            "purple": [1,0,1],
            "cyan": [0,1,1],
            "pink": [1,0,0.5],
        '''
        msg = String()
        msg.data = dir
        self.led_pattern(msg)


if __name__ == "__main__":
    node = LaneFollowNode("lanefollow_node")
    node.traverse_town()
