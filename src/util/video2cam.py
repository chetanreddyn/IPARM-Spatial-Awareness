import cv2

# Open the two video files or capture devices (use 0 or 1 for webcam feeds)
video_path_1 = 4  # Replace with the first video file path or use a webcam index like 0
video_path_2 = 6 # Replace with the second video file path or use a webcam index like 1

cap1 = cv2.VideoCapture(video_path_1)
cap2 = cv2.VideoCapture(video_path_2)

# Check if videos opened successfully
if not cap1.isOpened():
    print("Error: Could not open video 1.")
    exit()
if not cap2.isOpened():
    print("Error: Could not open video 2.")
    exit()

# Read and display the video frames from both feeds
while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    # Break the loop if either video feed ends
    if not ret1 or not ret2:
        print("Reached the end of one of the videos or error reading frames.")
        break

    # Display the frames in two separate windows
    cv2.imshow('Video 1', frame1)
    cv2.imshow('Video 2', frame2)

    # Press 'q' to exit the video display
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video capture objects and close all windows
cap1.release()
cap2.release()
cv2.destroyAllWindows()
