import cv2
import dlib
import numpy as np
from face_warp import get_landmarks
from scipy.spatial import Delaunay

# Initialize webcam (0 = default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam successfully opened. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Display the frame
    cv2.imshow("Webcam Feed", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()

# locate the face using get_landmarks from face_warp.py
get_landmarks(frame)

#loctate hand using cv2 and numpy, ignoring the face


#track movement of the hands
