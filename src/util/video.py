import cv2
import argparse

# Set up argument parser to take video path as input
parser = argparse.ArgumentParser(description='Video file or webcam stream display.')
parser.add_argument('--video', type=str, default='0', help='Path to video file or webcam (default: 0 for webcam)')
args = parser.parse_args()

# Convert video path argument to integer if it's for a webcam
video_path = int(args.video) if args.video.isdigit() else args.video

cap = cv2.VideoCapture(video_path)



# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Read and display the video frame by frame
while True:
    ret, frame = cap.read()
    
    # If frame read is successful, ret will be True
    if not ret:
        print("Reached the end of the video or error reading frame.")
        break
    
    # Display the frame
    cv2.imshow('Video Cam {}'.format(video_path), frame)
    
    # Press 'q' to exit the video display
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
