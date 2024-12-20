#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge


class StereoCalibrationNode:
    def __init__(self,config_dict ):
        rospy.init_node('stereo_calibration_node')

        # Parameters
        self.checkerboard_size = config_dict["checkerboard_size"] #(8, 6)
        self.square_size = config_dict["square_size"] # 0.025
        self.camera_num1 = config_dict["camera_num1"]
        self.camera_num2 = config_dict["camera_num2"]
        self.bridge = CvBridge()

        # Initialize variables
        self.cam_1_matrix = None
        self.cam_1_dist_coeffs = None
        self.cam_2_matrix = None
        self.cam_2_dist_coeffs = None

        # Subscriptions
        rospy.Subscriber('/multicam/cam_1/image_rect', Image, self.cam_1_callback)
        rospy.Subscriber('/multicam/cam_2/image_rect', Image, self.cam_2_callback)
        rospy.Subscriber('/multicam/cam_1/camera_info', CameraInfo, self.cam_1_info_callback)
        rospy.Subscriber('/multicam/cam_2/camera_info', CameraInfo, self.cam_2_info_callback)

        # Initialize variables for images
        self.cam_1_image = None
        self.cam_2_image = None

    def cam_1_callback(self, msg):
        self.cam_1_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def cam_2_callback(self, msg):
        self.cam_2_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def cam_1_info_callback(self, msg):
        self.cam_1_matrix = np.array(msg.K).reshape(3, 3)
        # rospy.loginfo("Cam1 K:\n%s",np.round(self.cam_1_matrix,2))
        self.cam_1_dist_coeffs = np.array(msg.D)
        # rospy.loginfo("Cam1 D:\n%s",np.round(self.cam_1_dist_coeffs,2))


    def cam_2_info_callback(self, msg):
        self.cam_2_matrix = np.array(msg.K).reshape(3, 3)
        # rospy.loginfo("Cam2 K:\n%s",np.round(self.cam_2_matrix,2))

        self.cam_2_dist_coeffs = np.array(msg.D)
        # rospy.loginfo("Cam2 D:\n%s",np.round(self.cam_2_dist_coeffs,2))


    def find_transformation_matrix(self):
        if self.cam_1_image is None or self.cam_2_image is None:
            rospy.loginfo("Waiting for images...")
            return

        if self.cam_1_matrix is None or self.cam_2_matrix is None:
            rospy.loginfo("Waiting for camera info...")
            return

        # Prepare object points (checkerboard grid in 3D space)
        objp = np.zeros((np.prod(self.checkerboard_size), 3), dtype=np.float32) # Shape of 48,3
        objp[:, :2] = np.mgrid[0:self.checkerboard_size[0], 0:self.checkerboard_size[1]].T.reshape(-1, 2)
        objp *= self.square_size

        # Find corners in cam_1
        gray_cam_1 = cv2.cvtColor(self.cam_1_image, cv2.COLOR_BGR2GRAY)
        ret_cam_1, corners_cam_1 = cv2.findChessboardCorners(gray_cam_1, self.checkerboard_size)

        # Find corners in cam_2
        gray_cam_2 = cv2.cvtColor(self.cam_2_image, cv2.COLOR_BGR2GRAY)
        ret_cam_2, corners_cam_2 = cv2.findChessboardCorners(gray_cam_2, self.checkerboard_size)

        if ret_cam_1 and ret_cam_2:
            # Refine corner locations
            # corners_cam_1 = cv2.cornerSubPix(
            #     gray_cam_1, corners_cam_1, (11, 11), (-1, -1),
            #     criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            # )
            # corners_cam_2 = cv2.cornerSubPix(
            #     gray_cam_2, corners_cam_2, (11, 11), (-1, -1),
            #     criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            # )

            # Collect points for stereo calibration
            objpoints = [objp]  # 3D points
            imgpoints_cam_1 = [corners_cam_1]  # 2D points from camera 1
            imgpoints_cam_2 = [corners_cam_2]  # 2D points from camera 2

            # Perform stereo calibration
            retval, cam_1_matrix, cam_1_dist_coeffs, cam_2_matrix, cam_2_dist_coeffs, R, T, E, F = cv2.stereoCalibrate(
                objpoints, imgpoints_cam_1, imgpoints_cam_2,
                self.cam_1_matrix, self.cam_1_dist_coeffs,
                self.cam_2_matrix, self.cam_2_dist_coeffs,
                gray_cam_1.shape[::-1],
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6),
                flags=cv2.CALIB_FIX_INTRINSIC
            )

            if retval:
                rospy.loginfo("Rotation matrix (R):\n%s",np.round(R,3))
                rospy.loginfo("Translation vector (T):\n%s",np.round(T,3))
                # rospy.loginfo("Essential matrix (E):\n{E}")
                # rospy.loginfo("Fundamental matrix (F):\n{F}")
            else:
                rospy.logerr("Stereo calibration failed.")
        else:
            rospy.logwarn("Checkerboard not found in both images.")

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            self.find_transformation_matrix()
            rate.sleep()


if __name__ == '__main__':
    try:

        config_dict = {"checkerboard_size": (8,6),
                       "square_size":0.025,
                       "camera_num1":1,
                       "camera_num2":2}

        node = StereoCalibrationNode(config_dict)
        node.run()
    except rospy.ROSInterruptException:
        pass
