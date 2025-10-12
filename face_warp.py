# Face warp code
# TODO: Later checken wat hiervan nodig is. Mediapipe is AI, dlib is toegestaan
# dlib heeft cmake & VS installer C++ nodig.
# zet interpreter global

# TODO: Check packages
import argparse
import sys
import os 
import json
import cv2
import numpy as np
import dlib
import matplotlib as plt # might be necessary
from skimage.io import imread

cap = cv2.VideoCapture(0)  # Webcam
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Helper functions
def get_landmarks(predictor, gray, face):
    landmarks = predictor(gray, face)
    landmarks_points = []
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmarks_points.append((x, y))
    return landmarks_points

def apply_affine_transform(src, src_tri, dst_tri, size):
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
    dst = cv2.warpAffine(src, warp_mat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return dst

# Code below used to track face in real-time and warp face to a square.
while True:
    ret, frame = cap.read()
    if not ret:
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)

    scale = frame.shape[1] / small_frame.shape[1]
    for face in faces:
        x1, y1 = face.left()*2, face.top()*2
        x2, y2 = face.right()*2, face.bottom()*2
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Live Face", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



