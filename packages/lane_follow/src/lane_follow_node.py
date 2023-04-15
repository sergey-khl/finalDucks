#!/usr/bin/env python3

import cv2
import numpy as np
import rospy
import yaml
import re

from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CameraInfo, CompressedImage
from std_msgs.msg import Float32, String, Bool
from turbojpeg import TurboJPEG
from geometry_msgs.msg import Point
from duckietown_msgs.msg import WheelsCmdStamped, Twist2DStamped
import tf.transformations as tft
from nav_msgs.msg import Odometry


# Define the HSV color range for road and stop mask
ROAD_MASK = [(20, 60, 0), (50, 255, 255)]
#STOP_MASK = [(0, 120, 120), (15, 255, 255)]
STOP_MASK = [(0, 55, 140), (20, 255, 255)]
CROSS_MASK = [(95, 120, 110), (135, 255, 210)]

# Parking lot values 
PARKING_LOT = {1:'207', 2:'226', 3:'228', 4:'75'}
ENTRANCE = '227' # need to change 

# Turn pattern 
TURN_VALUES = {'S': 0, 'L': np.pi/2, 'R': -np.pi/2}
TURN_RADIUS = {"S": 0, "L": 0.3, "R": 0.12}
STOP_RED = False
STOP_BLUE = False
STOP_BROKEN = False

# Set debugging mode (change to True to enable)
DEBUG = False

class LaneFollowNode(DTROS):
    def __init__(self, node_name):
        super(LaneFollowNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

        # Save node name and vehicle name
        self.node_name = node_name
        self.veh = rospy.get_param("~veh")
        self.park = int(rospy.get_param("~park"))
        print(self.park)
        self.parking_info = None

        # Publishers & Subscribers
        self.pub = rospy.Publisher("/" + self.veh + "/output/image/mask/compressed",
                                   CompressedImage,
                                   queue_size=1)

        self.sub = rospy.Subscriber("/" + self.veh + "/camera_node/image/compressed",
                                    CompressedImage,
                                    self.callback,
                                    queue_size=1,
                                    buff_size="20MB")
                                    
        self.sub_turn = rospy.Subscriber("/" + self.veh + "/turn",
                                    String,
                                    self.cb_turn,
                                    queue_size=1)

        self.sub_dist = rospy.Subscriber("/" + self.veh + "/duckiebot_distance_node/distance", Point, self.cb_dist, queue_size=1)   

        # Initialize distance subscriber and velocity publisher
        self.vel_pub = rospy.Publisher("/" + self.veh + "/car_cmd_switch_node/cmd",
                                       Twist2DStamped,
                                       queue_size=1)
        
        # Initialize TurboJPEG decoder
        self.jpeg = TurboJPEG()
        
        self.duck_gone = True


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

        self.latest_turn = "S"
        self.stopped_for_broken = False

        # Robot Pose Variables
        self.displacement = 0
        self.orientation = 0
        
        self.subscriber = rospy.Subscriber(f'/{self.veh}/deadreckoning_node/odom', 
                                           Odometry, 
                                           self.odom_callback)
                                           
        # Initialize shutdown hook
        rospy.on_shutdown(self.hook)


    def color_mask(self, img, mask, crop_width, pid=False, stopping=False, crossing=False, detect_duck=False):
        global STOP_RED, STOP_BLUE, DEBUG
        if crossing or stopping:
            img = img[:, 200:-200, :]
        # Convert the cropped image to HSV color space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Apply the color mask to the HSV image
        mask = cv2.inRange(hsv, mask[0], mask[1])
        crop = cv2.bitwise_and(img, img, mask=mask)

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
                    #print('stoping cond cy: ', cy, cx)
                    #self.proportional_stopline = (cy/168)*0.15

                    if cy >= 140:
                        STOP_RED = True

                # If checking for stopping condition or below the  and cx in range(100, 200)threshold, set STOP_RED
                elif crossing:
                    #print('crossing cond cy: ', cy, cx)
                    #self.proportional_stopline = (cy/168)*0.12

                    if cy >= 140:
                        STOP_BLUE = True

                if detect_duck:
                    self.duck_gone = False


                # Draw the contour and centroid on the image (for debugging)
                if DEBUG:
                    cv2.drawContours(crop, contours_road, max_idx, (0, 255, 0), 3)
                    cv2.circle(crop, (cx, cy), 7, (0, 0, 255), -1)
                    if detect_duck:
                        self.pub.publish(CompressedImage(format="jpeg", data=cv2.imencode('.jpg', crop)[1].tobytes()))
            except:
                pass

        # If no contour is found, reset the flags
        else:
            if pid:
                self.proportional = None

            elif stopping:
                self.proportional_stopline = None
                STOP_RED = False
                self.duck_gone = True
            elif crossing:
                self.proportional_stopline = None
                STOP_BLUE = False


    def callback(self, msg):
        # Decode the JPEG image from the message
        img = self.jpeg.decode(msg.data)
        h, w, _ = img.shape
        # Crop the image to focus on the road
        crop_road = img[250:-1, :, :]

        crop_width = crop_road.shape[1]

        # Process the image for PID control using the ROAD_MASK
        self.color_mask(crop_road, ROAD_MASK, crop_width, pid=True)

        # Process the image for stopping condition using the STOP_MASK
        self.color_mask(crop_road, STOP_MASK, crop_width, stopping=True, detect_duck=True)

        # Process the image for ducks using the DUCK_MASK
        self.color_mask(crop_road, CROSS_MASK, crop_width, crossing=True)


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
        self.displacement = displacement # broken :(



    def cb_dist(self, msg):        
        global STOP_BROKEN
        # If the z-value is less than 0.5, set stopping to True
        if (msg.z < 0.5 and msg.z != 0):
            STOP_BROKEN = True


    def turn(self, r, target_angle):

        # Calculate the target orientation 
        target_orientation = self.orientation + target_angle
        orientation_error_raw = target_orientation - self.orientation
        orientation_error = np.sign(orientation_error_raw) * (abs(orientation_error_raw) % (2*np.pi))

        # Create a rospy.Rate object to maintain the loop rate at 8 Hz
        rate = rospy.Rate(8)

        while abs(orientation_error) > 0.3:
            print('orientation_error: ', np.rad2deg(orientation_error))

            orientation_error_raw = target_orientation - self.orientation
            orientation_error = np.sign(orientation_error_raw) * (abs(orientation_error_raw) % (2*np.pi))

            # Calculate the angular speed using a simple controller
            # angular_speed = 0.296 * np.log(np.sign(orientation_error) * orientation_error + 0.06) +0.42
            if r == TURN_RADIUS["L"]:
                angular_speed = 3.2 * np.sign(orientation_error)
                linear_speed = 0.3

            elif r == 0: 
                if self.park in [3, 4]:
                    
                    angular_speed = -1*abs(5.1 * np.sign(orientation_error))
                    linear_speed = 0
                else:
                    angular_speed = abs(5.1 * np.sign(orientation_error))
                    linear_speed = 0

            else:
                angular_speed = 3.6 * np.sign(orientation_error)
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
                #print(self.proportional, P, D, self.twist.omega, self.twist.v)
                pass

        if self.proportional_stopline is None:
            self.twist.v = self.velocity

        else:
            self.twist.v = self.velocity - self.proportional_stopline

        self.vel_pub.publish(self.twist)


    def traverse_town(self):
        global STOP_RED, STOP_BLUE, STOP_BROKEN
        rate = rospy.Rate(8)  # 8hz

        while not rospy.is_shutdown():
            
            # Continue driving until stopline, or broken bot 
            while not STOP_RED and not STOP_BLUE and not STOP_BROKEN:
                self.drive()
                rate.sleep()

            latest_turn = self.latest_turn

            # Stop the Duckiebot once a sign is detected
            before_stop_red = STOP_RED
            if before_stop_red and latest_turn != "D":
                self.move_robust(speed=0 ,seconds=2)
                STOP_RED = before_stop_red

                print('STOP_RED', STOP_RED)
            
            before_stop_blue = STOP_BLUE
            if before_stop_blue:
                self.move_robust(speed=0 ,seconds=2)

                print('STOP_BLUE', before_stop_blue)

            before_stop_broken = STOP_BROKEN
            if before_stop_broken and not self.stopped_for_broken:
                self.move_robust(speed=0 ,seconds=2)

                print('STOP_BROKEN', before_stop_broken)
                self.stopped_for_broken = True
                
            print(latest_turn)

            # If a stop line is detected
            if STOP_RED and latest_turn != "D":
                # we are about to enter the parking lot
                if latest_turn == "P":
                    print('Time to park!')
                    self.parking_lot()
                
                elif latest_turn[0] == 'i':
                    stall_id, x, y, z = self.extract_parking_info()
                    if z > 1 and str(int(stall_id)) == ENTRANCE:
                        print('Time to park!')
                        self.parking_lot()
                        continue
                
                # at a regualr intersection about to make a turn
                else:
                    print(latest_turn)
                    turning_angle = TURN_VALUES[latest_turn]

                    if turning_angle == 0: # go straight 
                        self.move_robust(speed=0.3 ,seconds=2.5)
                        STOP_RED = False

                    else: # make turn 
                        self.turn(TURN_RADIUS[latest_turn], turning_angle)

                    STOP_RED = False

            elif STOP_BLUE and self.duck_gone:
                self.move_robust(speed=self.velocity ,seconds=2)
                STOP_BLUE = False
                self.duck_gone = False

            elif STOP_BROKEN and self.stopped_for_broken:
                STOP_BROKEN = False
                self.offset = -150
                #start = self.displacement.copy()
                
                rate = rospy.Rate(8)
                # while abs(self.displacement - start) < 0.25:
                #     print(start, self.displacement)
                    # # Publish the twist message multiple times to ensure the robot stops
                for i in range(int(32)):
                    self.drive()
                    rate.sleep()

                STOP_BROKEN = False
                
                self.offset = 200

                # for i in range(int(20)):
                #     self.drive()
                #     rate.sleep()

                #self.P = 0.035


    def april_tag_PID(self, tag_id, threshold):
        made_it = False

        rate = rospy.Rate(8)

        while not made_it:
            if self.parking_info == None:
                print('parking_info is none')
                continue
            try:
                stall_id, x, y, z = self.extract_parking_info()
            except:
                print('too many vars')

            if tag_id == str(int(stall_id)):
                print('x: ', x, 'y: ', y, 'z: ', z)

                if z < threshold:
                    made_it = True

                # need to change these 
                self.twist.v, self.twist.omega = z*0.25+0.1, 0 #-1*(190-x)*1/10 # is it x or y? mul by -1?
                self.vel_pub.publish(self.twist)

            rate.sleep()


    def extract_parking_info(self):
        pattern = r'-?\d+(\.\d+)?'

        matches = []
        for match in re.finditer(pattern, self.parking_info):
            number = float(match[0])
            matches.append(number if number.is_integer() else number)

        return matches


    def parking_lot(self):
        stopping_threshold_stall = 0.3
        stopping_threshold_entrance = 0.5

        # move to middle of parking lot
        self.april_tag_PID(ENTRANCE, stopping_threshold_entrance)

        print('made it to middle')
        self.move_robust(speed=0 ,seconds=2)

        # turn to correct direction
        if self.park in [1,2]:
            self.turn(0, np.pi/2)
        else:
            self.turn(0, -1*np.pi/2)
        
        # ok lets park!
        desired_id = PARKING_LOT[self.park]
        self.april_tag_PID(desired_id, stopping_threshold_stall)

        rospy.signal_shutdown('Parked!')


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


    def cb_turn(self, turn):
        print('latest, ', turn.data)

        if turn.data[0] == "i":
            self.parking_info = turn.data

        self.latest_turn = turn.data


    def hook(self):
        print("SHUTTING DOWN")
        self.move_robust(0, 3)


if __name__ == "__main__":
    try:
        node = LaneFollowNode("lanefollow_node")
        node.traverse_town()
    except rospy.ROSInterruptException:
        pass