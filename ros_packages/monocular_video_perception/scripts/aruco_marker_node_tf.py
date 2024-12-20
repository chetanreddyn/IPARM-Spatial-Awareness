#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import os
import yaml
import tf
import rospkg
import argparse
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
        self.camera_num = config_dict["camera_num"]
        self.translation_buffer = np.zeros((config_dict["buffer_size"],3))

        self.quaternion_buffer = np.zeros((config_dict["buffer_size"],4))
        self.quaternion_buffer[:,0] = 1 # Making all elements (1,0,0,0)
        # self.rotation_buffer = np.zeros()
    
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

    def trans_quaternion_to_matrix(self, transform):
        """
        Convert a transform message to a 4x4 matrix.
        Transform is a tuple (trans,rot), rot is a quarternion of shape 4
        """
        translation = transform[0]
        rotation_quaternion = transform[1]
        R_44 = np.eye(4)

        # Extract quaternion
        # quaternion = [rotation.x, rotation.y, rotation.z, rotation.w]
        rotation_matrix = tf.transformations.quaternion_matrix(rotation_quaternion)
        R_44[:3, :3] = rotation_matrix[:3, :3]

        # Fill translation part
        R_44[0, 3] = translation[0]
        R_44[1, 3] = translation[1]
        R_44[2, 3] = translation[2]

        return R_44
    
    def compute_inverse(self,matrix):
        """
        Uses the formula T_inv = [RT -RT*t;0 1]
        """
        R = matrix[:3,:3]
        t = matrix[:3,3:4]
        T_inv = np.eye(4)
        T_inv[:3,:3] = R.T
        T_inv[:3,3:4] = -(R.T)@t
        return T_inv
    
    def matrix_to_trans_quarternion(self,matrix):
        translation = matrix[:3, 3]
        quaternion = tf.transformations.quaternion_from_matrix(matrix)
        return translation,quaternion
    
    def average_quaternions(self,quaternions):
        """
        Average a set of quaternions.

        Parameters:
        quaternions (numpy.ndarray): Array of shape (N, 4), where N is the number of quaternions.
                                    Each quaternion is represented as [w, x, y, z].

        Returns:
        numpy.ndarray: Averaged quaternion of shape (4,).
        """
        # Validate input dimensions
        if quaternions.ndim != 2 or quaternions.shape[1] != 4:
            raise ValueError("Input array must have shape (N, 4).")

        # Normalize each quaternion to ensure they are valid unit quaternions
        quaternions = quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True)

        # Create a symmetric accumulator matrix (4x4) to sum the outer products
        accumulator = np.zeros((4, 4))
        for q in quaternions:
            accumulator += np.outer(q, q)

        # Scale the accumulator matrix
        accumulator /= len(quaternions)

        # Compute the eigenvalues and eigenvectors of the accumulator matrix
        eigenvalues, eigenvectors = np.linalg.eig(accumulator)

        # The eigenvector corresponding to the largest eigenvalue is the average quaternion
        avg_quaternion = eigenvectors[:, np.argmax(eigenvalues)]

        # Ensure the quaternion has a positive scalar part (w) for consistency
        if avg_quaternion[0] < 0:
            avg_quaternion = -avg_quaternion

        return avg_quaternion
    
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
                # rvec and tvec represent the orientation and position of aruco wrt camera

                # If pose estimation is successful, publish the transform
                if ret:
                    # Create the transformation from the camera frame to the marker's frame
                    marker_frame = f"aruco_frame_cam_{self.camera_num}" # Parent Frame since c1Tm goes from marker to c1
                    
                    # Convert rotation vector (rvec) to a rotation matrix
                    R, _ = cv2.Rodrigues(rvec)

                    translation_raw = tvec.flatten()
                    self.translation_buffer[1:,:] = self.translation_buffer[:-1,:]
                    self.translation_buffer[0,:] = translation_raw

                    translation_avg = self.translation_buffer.mean(axis=0)
                    # Create a transform broadcaster
                    R_44 = np.eye(4)
                    R_44[:3, :3] = R  # Fill the rotation part

                    quaternion_raw = tf.transformations.quaternion_from_matrix(R_44) 
                    self.quaternion_buffer[1:,:] = self.quaternion_buffer[:-1,:]
                    self.quaternion_buffer[0,:] = quaternion_raw
                    quaternion_avg = self.average_quaternions(self.quaternion_buffer)

                    aruco_to_cam_matrix = self.trans_quaternion_to_matrix((translation_avg,quaternion_avg))
                    cam_to_aruco_matrix = self.compute_inverse(aruco_to_cam_matrix)
                    translation_avg_inv, quaternion_avg_inv = self.matrix_to_trans_quarternion(cam_to_aruco_matrix)


                    child_frame = marker_frame # Camera is child frame c1Tm
                    parent_frame = "/multicam/cam_{}/camera_frame".format(self.camera_num) 
                    broadcaster.sendTransform(
                        translation_avg_inv,  # translation
                        quaternion_avg_inv,      # 4x4 rotation matrix as input 
                        rospy.Time.now(),                                   # timestamp
                        child_frame,                                       # child frame
                        parent_frame                    # parent frame
                    )
                    # print("Broadcasted from {} to {}".format(marker_frame,child_frame))
                    rospy.loginfo("Transform from aruco to cam matrix:\n%s", np.round(self.trans_quaternion_to_matrix((translation_avg,quaternion_avg)),2))
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
        self.camera_num = config_dict["camera_num"]
        
        # Create ROS publisher and subscriber
        self.image_pub = rospy.Publisher('/multicam/cam_{}/image_annotated'.format(self.camera_num), Image, queue_size=10)
        self.image_sub = rospy.Subscriber('/multicam/cam_{}/image_rect'.format(self.camera_num), Image, self.callback)
        
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

    parser = argparse.ArgumentParser(description="Aruco Marker Detection")
    parser.add_argument('--camera_num', type=int, default=1, help="Camera number (default is 1)")
    args,_ = parser.parse_known_args()

    camera_num = args.camera_num 
    # camera_num = rospy.get_param('camera_num')
    print("CAMERA NUM: ",camera_num)

    rospy.init_node('aruco_marker_detection_cam_{}'.format(camera_num), anonymous=True)
    
    
    # Path to the calibration file
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('monocular_video_perception')

    # Path to the calibration file
    yaml_file = os.path.join(package_path, "calibration_files", "cam_{}.yaml".format(camera_num))

    config_dict = {"marker_size": 0.09, # Marker size in meters (e.g., 2.1cm)
                   "camera_num":camera_num,
                   "buffer_size":10}  

    # Create and run the ArUco marker node
    aruco_node = ArUcoMarkerNode(yaml_file, config_dict)

    # Keep the node running
    rospy.spin()


if __name__ == '__main__':
    main()
