Project: Spongebob filter after making a rainbow gesture
=================================================

Description:
-------------
This project implements real-time face and hand tracking. When the user performs a rainbow gesture ober their head, the face is warped, gets a yellow tint, and brown spots appear on the face (like SpongeBob). Besdies that a rainbow appears and the text imagination appears (based on the meme).

What it does:
----------
- Face detection and 68-point landmarks (dlib)
   - plus adding 9 more landmarks to recognise the forehead as well
- Hand detection and tracking (MediaPipe Hands)
- Rainbow gesture detection
- Face warping
- Yellow tint overlay
- Brown spots on the face
- Adding ranbow image and text

Requirements for the script:
-------------
The following libraries:
- OpenCV
- Dlib 
- Mediapipe 
- NumPy 
- SciPy (Delaunay)
- collections (deque)

png of the rainbow to add in


How to use it:
------
1. Run the main script: finalFilter
   It uses the script functions, in which all the functions of the actual implementation are located.


2. Perform the rainbow gesture:
   - start is hands close above the head
   -  move hands apart around the sides
   - End position: one hand next to the head on either side

3. Press 'q' to quit.

Files:
------
- finalFilter  (main loop)
- functions (all implementation functions)

