#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import os
import yaml
import tf
import rospkg
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from cv2 import aruco

# Existing CameraCalibration_and_ArUcoPoseEstimator class
class CameraCalibration_and_ArUcoPoseEstimator:
    def __init__(self, yaml_file, config_dict):
        self.yaml_file = yaml_file
        self.camera_matrix = None
        self.dist_coeffs = None
        self.load_calibration()
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        self.parameters = cv2.aruco.DetectorParameters_create()
        self.marker_size = config_dict["marker_size"]  # Adjust this based on your marker size
    
    def load_calibration(self):
        """Load camera calibration data from a YAML file."""
        with open(self.yaml_file, 'r') as f:
            data = yaml.safe_load(f)
        
        self.camera_matrix = np.array(data['camera_matrix']['data']).reshape(3, 3)
        self.dist_coeffs = np.array(data['distortion_coefficients']['data'])
        
    def get_camera_matrix(self):
        return self.camera_matrix
    
    def get_dist_coeffs(self):
        return self.dist_coeffs

    def detect_and_estimate_pose(self, frame, broadcaster, marker_id_offset=0):
        """Detect ArUco markers and estimate their pose in real time."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)

        if len(corners) > 0:
            # Estimate pose for each detected marker
            for i in range(len(corners)):
                object_points = np.array([
                    [-self.marker_size / 2, self.marker_size / 2, 0],  # top-left
                    [self.marker_size / 2, self.marker_size / 2, 0],   # top-right
                    [self.marker_size / 2, -self.marker_size / 2, 0],  # bottom-right
                    [-self.marker_size / 2, -self.marker_size / 2, 0]  # bottom-left
                ], dtype=np.float32)

                image_points = corners[i].reshape(4, 2)  # For the first detected marker
                ret, rvec, tvec = cv2.solvePnP(object_points, image_points, self.camera_matrix, self.dist_coeffs)

                # If pose estimation is successful, publish the transform
                if ret:
                    # Create the transformation from the camera frame to the marker's frame
                    marker_frame = f"aruco_marker_{marker_id_offset + i}"
                    
                    # Convert rotation vector (rvec) to a rotation matrix
                    R, _ = cv2.Rodrigues(rvec)

                    translation = tvec.flatten()

                    # Create a transform broadcaster
                    R_44 = np.eye(4)
                    R_44[:3, :3] = R  # Fill the rotation part

                    broadcaster.sendTransform(
                        (translation[0], translation[1], translation[2]),  # translation
                        tf.transformations.quaternion_from_matrix(R_44),      # 4x4 rotation matrix as input 
                        rospy.Time.now(),                                   # timestamp
                        marker_frame,                                       # child frame
                        "/multicam/cam1/camera_frame"                        # parent frame
                    )
                    print("Broatcasted to {}".format(marker_frame))
                    # Visualize 3D axes on the frame
                    cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.1)  # 0.1 is the length of the axes

                    # Print and write the ID and the X, Y, Z coordinates of the marker on the frame
                    if ids is not None and ids[i] is not None:
                        marker_id = ids[i][0]  # Shape of ids is (num_markers,1)
                        x, y, z = tvec[:, 0]  # Shape is (3,1)
                        
                        # Convert position values to text
                        text = f"ID: {marker_id} X: {x:.2f} Y: {y:.2f} Z: {z:.2f}"
                        
                        # Define position to write the text on the frame (above the marker)
                        position = (int(corners[i][0][0][0]), int(corners[i][0][0][1] - 10))  # Slightly above the top-left corner of the marker
                        
                        # Use cv2.putText to draw the text on the frame
                        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            
        return frame


class ArUcoMarkerNode:
    def __init__(self, yaml_file, config_dict):
        self.pose_estimator = CameraCalibration_and_ArUcoPoseEstimator(yaml_file, config_dict)
        self.bridge = CvBridge()
        
        # Create ROS publisher and subscriber
        self.image_pub = rospy.Publisher('/multicam/cam1/image_annotated', Image, queue_size=10)
        self.image_sub = rospy.Subscriber('/multicam/cam1/image_raw', Image, self.callback)
        
        # Initialize TF broadcaster
        self.tf_broadcaster = tf.TransformBroadcaster()
    
    def callback(self, msg):
        # Convert the ROS image message to a cv2 image
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
        # Detect and estimate pose, and publish the transforms
        frame_annotated = self.pose_estimator.detect_and_estimate_pose(frame, self.tf_broadcaster)

        # Convert the annotated frame back to ROS image message
        annotated_msg = self.bridge.cv2_to_imgmsg(frame_annotated, "bgr8")

        # Publish the annotated image
        self.image_pub.publish(annotated_msg)


def main():
    # Initialize the ROS node
    rospy.init_node('aruco_marker_detection', anonymous=True)
    
    # Path to the calibration file
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('monocular_video_perception')

    # Path to the calibration file
    yaml_file = os.path.join(package_path, "calibration_files", "cam1.yaml")

    config_dict = {"marker_size": 0.09}  # Marker size in meters (e.g., 9cm)

    # Create and run the ArUco marker node
    aruco_node = ArUcoMarkerNode(yaml_file, config_dict)

    # Keep the node running
    rospy.spin()


if __name__ == '__main__':
    main()
