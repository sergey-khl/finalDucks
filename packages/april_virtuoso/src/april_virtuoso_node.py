#!/usr/bin/env python3
import os
from pathlib import Path
import rospy
import cv2
import numpy as np

from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CameraInfo, CompressedImage
from std_msgs.msg import String, Int32, Bool
from nav_msgs.msg import Odometry
from turbojpeg import TurboJPEG
import yaml
from lane_follow.srv import img


NUM_MASK = [(90, 85, 100), (175, 255, 255)]

class AprilVirtuosoNode(DTROS):

    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(DigitVirtuosoNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        self.veh = rospy.get_param("~veh")
        
        self.pub = rospy.Publisher("/" + self.veh + "/output/image/mask/compressed",
                                   CompressedImage,
                                   queue_size=1)
                                   
        self.pub_done = rospy.Publisher(f'/{self.veh}/done', Bool, queue_size=1)
       
        self.sub = rospy.Subscriber("/" + self.veh + "/camera_node/image/compressed",
                                    CompressedImage,
                                    self.callback,
                                    queue_size=1,
                                    buff_size="20MB")

        # setup publisher
        # Initialize TurboJPEG decoder
        self.jpeg = TurboJPEG()
        self.undistorted = None

        self.calibration_file = f'/data/config/calibrations/camera_intrinsic/default.yaml'
 
        self.calibration = self.readYamlFile(self.calibration_file)

        self.img_width = self.calibration['image_width']
        self.img_height = self.calibration['image_height']
        self.cam_matrix = np.array(self.calibration['camera_matrix']['data']).reshape((self.calibration['camera_matrix']['rows'], self.calibration['camera_matrix']['cols']))
        self.distort_coeff = np.array(self.calibration['distortion_coefficients']['data']).reshape((self.calibration['distortion_coefficients']['rows'], self.calibration['distortion_coefficients']['cols']))

        self.new_cam_matrix, self.roi = cv2.getOptimalNewCameraMatrix(self.cam_matrix, self.distort_coeff, (self.img_width, self.img_height), 1, (self.img_width, self.img_height))
        
        # Initialize get april tag service
        april_service = f'/{self.veh}/augmented_reality_node/get_april_detect'
        rospy.wait_for_service(april_service)
        self.get_april = rospy.ServiceProxy(april_service, img)

        # Initialize get digit service
        digit_service = f'/detect_digit_node/detect_digit'
        rospy.wait_for_service(digit_service)
        self.get_digit = rospy.ServiceProxy(digit_service, img)
        
        self.found = []
        self.done = False

    def callback(self, msg):
        # decode the received message to an image
        img = self.jpeg.decode(msg.data)

        # https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
        undistorted = cv2.undistort(img, self.cam_matrix, self.distort_coeff, None, self.new_cam_matrix)
        x, y, w, h = self.roi
        undistorted = undistorted[y:y+h, x:x+w]
        crop = self.crop_num(undistorted)
        if crop is not False:
 
            processed = self.mask_num(crop)
            processed = cv2.resize(processed, (28, 28))
            self.masked = CompressedImage(format="jpeg", data=cv2.imencode('.jpg', processed)[1].tobytes()) if processed is not False else None

            self.undistorted = CompressedImage(format="jpeg", data=cv2.imencode('.jpg', undistorted)[1].tobytes())
        
            if self.masked is not None:
            
                self.pub.publish(self.masked)
                if not cv2.imwrite(f'/data/test/{rospy.Time.now().secs}.png', processed):
                    raise Exception("Could not write image")
                if not cv2.imwrite(f'/data/crop/{rospy.Time.now().secs}.png', crop):
                    raise Exception("Could not write image")
                if not cv2.imwrite(f'/data/normal/{rospy.Time.now().secs}.png', undistorted):
                    raise Exception("Could not write image")
        
    def crop_num(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Apply the color mask to the HSV image
        mask = cv2.inRange(hsv, NUM_MASK[0], NUM_MASK[1])
        crop = cv2.bitwise_and(img, img, mask=mask)

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

        if len(contours_road) > 0:
            x, y, w, h = cv2.boundingRect(contours_road[max_idx])
        else:
            return False

        return img[y:y+h, x:x+w]

    def mask_num(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply a threshold using the histogram approach (e.g., Otsu's method)
        #_, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        thresholded = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)

        # Remove noise
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the contour with the maximum area
        max_area = 0
        max_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                max_contour = contour

        # Create a black mask with the same size as the opening image
        mask = np.zeros(opening.shape, dtype=np.uint8)
        
        if max_contour is not None:

            cv2.drawContours(mask, [max_contour], 0, 255, -1)
        else:
            return False

        # Apply the mask to the original image
        masked_image = cv2.bitwise_and(opening, opening, mask=mask)
        return masked_image
        
    def maestro(self):
        if len(self.found) == 10:
            self.pub_done.publish(Bool(data=True))
            
        if self.undistorted is not None and self.masked is not None:
            num = self.get_digit(self.masked).found.data
            if num not in self.found:
                self.found.append(num)
                if num != -1:
                    self.log(f"FOUND NUMBER: {num}")
                    detected = self.get_april(self.undistorted)

        
    def readYamlFile(self,fname):
        """
        Reads the YAML file in the path specified by 'fname'.
        E.G. :
            the calibration file is located in : `/data/config/calibrations/filename/DUCKIEBOT_NAME.yaml`
        """
        with open(fname, 'r') as in_file:
            try:
                yaml_dict = yaml.load(in_file)
                return yaml_dict
            except yaml.YAMLError as exc:
                self.log("YAML syntax error. File: %s fname. Exc: %s"
                        %(fname, exc), type='fatal')
                rospy.signal_shutdown()
                return


if __name__ == '__main__':
    # create the node
    node = AprilVirtuosoNode(node_name='april_virtuoso_node')
    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        node.maestro()
        rate.sleep()
