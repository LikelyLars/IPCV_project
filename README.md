Face & Hand Tracking with Rainbow Gesture Effects
=================================================

Description:
-------------
This project implements real-time face and hand tracking. When the user performs a "rainbow over the head" gesture, the face is warped, gets a yellow tint, and brown spots appear on the face (like SpongeBob texture). Yellow and brown tints can be adjusted using sliders.

Features:
----------
- Face detection and 68-point landmarks (dlib)
- Hand detection and tracking (MediaPipe Hands)
- Rainbow gesture detection
- Face warping and exaggeration
- Yellow tint overlay
- Brown spots on the face
- Real-time sliders for adjusting colors

Requirements:
-------------
- Python 3.8+
- OpenCV (`opencv-python`)
- Dlib (`dlib`)
- Mediapipe (`mediapipe`)
- NumPy (`numpy`)
- SciPy (`scipy`)
- `shape_predictor_68_face_landmarks.dat` (dlib model)
- `face_augment_2.py` with warp/exaggeration functions

Usage:
------
1. Run the script:
   python hand_tracking.py

2. Perform the "rainbow over the head" gesture:
   - Hands close above the head
   - Move hands apart around the sides

3. Adjust sliders in the "Tints" window:
   - Yellow Strength
   - Brown A Strength
   - Brown B Strength

4. Press 'q' to quit.

Notes:
------
- Only the first detected face is processed
- Sliders allow dynamic tuning for cartoonish effects
- Brown spots and yellow tint are smoothly blended

Files:
------
- hand_tracking.py  (main script)
- face_augment_2.py (helper functions for warping)
- shape_predictor_68_face_landmarks.dat (dlib model)
