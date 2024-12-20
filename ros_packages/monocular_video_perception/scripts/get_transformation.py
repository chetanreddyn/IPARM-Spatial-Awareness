#!/usr/bin/env python3
import rospy
# import tf2_ros
# import tf2_geometry_msgs
from geometry_msgs.msg import TransformStamped
import tf
import numpy as np
import argparse

class TransformProcessor:
    def __init__(self,config_dict):

        self.camera_num1 = config_dict["camera_num1"]
        self.camera_num2 = config_dict["camera_num2"]
        self.transform_age_thresh = config_dict["transform_age_thresh"]
        self.period_between_callbacks = config_dict["period_between_callbacks"]
        # rospy.init_node('transform_listener')

        # Initialize tf2 buffer and listener
        # self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf.TransformListener()
        self.tf_broadcaster = tf.TransformBroadcaster()
        self.translation_buffer = np.zeros((config_dict["buffer_size"],3))
        self.quaternion_buffer = np.zeros((config_dict["buffer_size"],4))
        self.quaternion_buffer[:,0] = 1 # Making all elements (1,0,0,0)

        self.timer = rospy.Timer(rospy.Duration(self.period_between_callbacks), self.print_transform_callback)  
    

    def lookup_transform(self, source_frame, target_frame):
        """
        Look up the transform from source_frame to target_frame. 
        Returns the transform if available. returns targetT_source
        """
        try:
            transform = self.tf_listener.lookupTransform(source_frame, target_frame, rospy.Time(0))
            # Transform is a tuple here with (trans,rot)

            # Check if the transform timestamp is recent enough (for example, within 1 second)
            current_time = rospy.Time.now()
            transform_time = self.tf_listener.getLatestCommonTime(target_frame, source_frame)
            transform_age = current_time - transform_time

            if transform_age.to_sec()<self.transform_age_thresh:
                return self.tf_listener.lookupTransform(source_frame, target_frame, rospy.Time(0))
            else:
                # rospy.logwarn(f"Transform from {source_frame} to {target_frame} is TOO OLD")
                return None
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            # rospy.logwarn(f"Waiting for transform from {source_frame} to {target_frame}...")
            return None

    def transform_to_matrix(self, transform):
        """
        Convert a tf2 transform message to a 4x4 matrix.
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

    def compute_transform(self, transform_aruco1_to_cam1, transform_aruco2_to_cam2):
        """
        Compute the transform from Cam2_frame to Cam1_frame.
        """
        # Convert transforms to 4x4 matrices
        transform_aruco1_to_cam1_matrix = self.transform_to_matrix(transform_aruco1_to_cam1)
        transform_aruco2_to_cam2_matrix = self.transform_to_matrix(transform_aruco2_to_cam2)

        # rospy.loginfo("Transform from aruco1 to Cam1_frame matrix:\n%s", np.round(transform_aruco1_to_cam1_matrix,2))

        # Compute the inverse of the transform from aruco_cam2_frame to cam2_frame
        transform_cam2_to_aruco2_matrix = np.linalg.pinv(transform_aruco2_to_cam2_matrix)
        transform_cam2_to_aruco2_matrix_custom = self.compute_inverse(transform_aruco2_to_cam2_matrix)

        # Now compute the transform from cam1_frame to cam2_frame
        transform_cam2_to_cam1_matrix = transform_aruco1_to_cam1_matrix@transform_cam2_to_aruco2_matrix

        # rospy.loginfo("Transform from aruco1 to Cam1_frame matrix:\n%s", np.round(transform_aruco1_to_cam1_matrix,2))
        # rospy.loginfo("Transform from aruco2 to Cam2_frame matrix:\n%s", np.round(transform_aruco2_to_cam2_matrix,2))
        # rospy.loginfo("Transform from cam2_frame to aruco2 matrix:\n%s", np.round(transform_cam2_to_aruco2_matrix_custom,2))
        # rospy.loginfo("Transform from cam2_frame to cam1 matrix:\n%s", np.round(transform_cam2_to_cam1_matrix,2))





        # rospy.loginfo("Transform fcustom:\n%s", np.round(transform_cam2_to_cam1_matrix,2))

        return transform_cam2_to_cam1_matrix

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

    def matrix_to_trans_quarternion(self,matrix):
        translation = matrix[:3, 3]
        quaternion = tf.transformations.quaternion_from_matrix(matrix)
        return translation,quaternion

    def broadcast_transform(self, matrix, child_frame, parent_frame):
        """
        Broadcast a transform given a 4x4 transformation matrix.
        """
        translation_raw = matrix[:3, 3]  # Extract translation
        quaternion_raw = tf.transformations.quaternion_from_matrix(matrix)  # Extract quaternion

        self.translation_buffer[1:,:] = self.translation_buffer[:-1,:]
        self.translation_buffer[0,:] = translation_raw

        self.quaternion_buffer[1:,:] = self.quaternion_buffer[:-1,:]
        self.quaternion_buffer[0,:] = quaternion_raw

        translation = self.translation_buffer.mean(axis=0)
        quaternion = self.average_quaternions(self.quaternion_buffer)

        euler_angles = np.array(tf.transformations.euler_from_quaternion(quaternion))#*180/np.pi
        # print(type(euler_angles))
        rospy.loginfo("Transform translation:%s", np.round(translation,2))
        rospy.loginfo("Transform rotation:%s", np.round(euler_angles,2))



        # # Create TransformStamped message
        # transform = TransformStamped()
        # transform.header.stamp = rospy.Time.now()
        # transform.header.frame_id = parent_frame
        # transform.child_frame_id = child_frame
        # transform.transform.translation.x = translation[0]
        # transform.transform.translation.y = translation[1]
        # transform.transform.translation.z = translation[2]
        # transform.transform.rotation.x = quaternion[0]
        # transform.transform.rotation.y = quaternion[1]
        # transform.transform.rotation.z = quaternion[2]
        # transform.transform.rotation.w = quaternion[3]

        # Broadcast the transform
        self.tf_broadcaster.sendTransform(translation,
                                          quaternion,
                                          rospy.Time.now(),
                                          child_frame,
                                          parent_frame)

    def print_transform_callback(self,event):
        # print(event)
        # rate = rospy.Rate(20.0)

        # while not rospy.is_shutdown():
            # Look up the transforms
            # First argument is target_frame and second argument is source frame
            # X_target = T @ X_source

        # cam1T_aruco1 represents pose of aruco1 in cam1
        transform_aruco1_to_cam1 = self.lookup_transform('aruco_frame_cam_{}'.format(self.camera_num1), 'multicam/cam_{}/camera_frame'.format(self.camera_num1))
        transform_aruco2_to_cam2 = self.lookup_transform('aruco_frame_cam_{}'.format(self.camera_num2), 'multicam/cam_{}/camera_frame'.format(self.camera_num2))

        if transform_aruco1_to_cam1 and transform_aruco2_to_cam2:
            print("Able to read the aruco from both frames!")
            # rospy.loginfo("Transform from Cam1_frame to aruco_cam1_frame: \n%s", transform_cam1_to_aruco1)
            # rospy.loginfo("Transform from Cam2_frame to aruco_cam2_frame: \n%s", transform_cam2_to_aruco2)

            # Calculate the transform between cam1_frame and cam2_frame
            transform_cam2_to_cam1_matrix = self.compute_transform(transform_aruco1_to_cam1, transform_aruco2_to_cam2)
            # Broadcast the computed transform using the broadcaster

            # child_frame = "multicam/cam_{}/camera_frame".format(self.camera_num1)
            # parent_frame = "multicam/cam_{}/camera_frame".format(self.camera_num2)
            child_frame = "multicam/cam_{}/test".format(self.camera_num1)
            parent_frame = "multicam/cam_{}/test".format(self.camera_num2)

            transform_cam1_to_cam2_matrix = self.compute_inverse(transform_cam2_to_cam1_matrix)
            self.broadcast_transform(transform_cam1_to_cam2_matrix,child_frame,parent_frame)
            print("Broadcasted transform from '{}' to '{}'".format(parent_frame,child_frame))
            

            # c1Tc2
            # rospy.loginfo("Transform from Cam2_frame to Cam1_frame matrix:\n%s", np.round(transform_cam2_to_cam1_matrix,2))

            # rate.sleep()

if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description="Aruco Marker Detection")
        parser.add_argument('--camera_num1', type=int, default=1, help="Camera number of first camera (default is 1)")
        parser.add_argument('--camera_num2', type=int, default=2, help="Camera number of second camera (default is 2)")

        args = parser.parse_args()
        camera_num1 = args.camera_num1
        camera_num2 = args.camera_num2

        rospy.init_node('camera_transform_c{}Tc{}'.format(camera_num1,camera_num2), anonymous=True)

        config_dict = {"camera_num1":camera_num1,
                       "camera_num2":camera_num2,
                       "transform_age_thresh":0.5,
                       "period_between_callbacks":0.1,
                       "buffer_size":50}
        
        transform_processor = TransformProcessor(config_dict)
        rospy.spin()
        # transform_processor.print_transform()
    except rospy.ROSInterruptException:
        pass
