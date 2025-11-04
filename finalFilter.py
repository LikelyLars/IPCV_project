import cv2
from functions import *

# ----- initialise videocapture ----- #
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print(" Error: Could not open webcam.")
    exit()

# ---- Initialise rainbow.png for filter overlay ------- #
rainbow_img = cv2.imread("rainbow.png", cv2.IMREAD_UNCHANGED)
rainbow_img = cv2.resize(rainbow_img, (600, 400))

# ----- Main loop ----- #
print("Press 'q' to quit.")


filter_on = False #bool to track if filter should be applied

while True:

    #reading image from webcam
    ret, input_frame = cap.read()
    if not ret:
        break

    #frame to be displayed
    display_frame = input_frame.copy()

    #frame to be processed (will get some corrections to improve tracking)
    process_frame = input_frame.copy()

    # apply lighting adjustment and gamma correction to improve consistency in hand & face tracking 
    process_frame = auto_gamma_correction(adjust_lighting(process_frame))

    # detect hands and update tracks
    detected_hands = detect_hands(process_frame)
    if detected_hands:
        tracks = update_hand_tracks(tracks, detected_hands)

    # detect faces and find landmarks
    faces = detect_faces(process_frame)


    if not faces:
        filter_on = False  # reset filter if no face is detected
        tracks = reset_hand_tracks() # reset hand tracks if no face is detected

    # check for rainbow gesture
    for face in faces:
        face_bbox = face["bbox"]
        landmarks = face["landmarks"]

        # ---- gesture detections
        if detect_rainbow_gesture(tracks, face_bbox):
            filter_on = True #apply filter
            generate_spots() #generate new sponge spots
            tracks = reset_hand_tracks() #reset handtracks to prevent immediate re-detection

        # ---- APPLY SPONGEBOB FILTER IF NEEDED ----
        if filter_on:
            
            # first warp the face and return new landmarks
            display_frame, landmarks = warp_face_region(display_frame, landmarks)

            # apply yellow color tint
            display_frame = apply_yellow_tint(display_frame, landmarks)

            # apply sponge spots
            display_frame = apply_brown_tint(display_frame, landmarks)

            # add rainbow overlay
            head_x, head_y, head_w, head_h = face_bbox

            # Desired position: center rainbow above the head, position slightly higher then the head top
            rainbow_h, rainbow_w = rainbow_img.shape[:2]
            overlay_x = head_x + head_w//2 - rainbow_w//2
            overlay_y = head_y - (rainbow_h - rainbow_h//5)  # place above the head

            #overlay rainbow image with alpha channel handling(only rainbow, not background)
            display_frame = overlay_image_alpha(display_frame, rainbow_img, overlay_x, overlay_y)

            # add "IMAGINATION" text (with correct flipping such that text is not mirrored in final output)
            display_frame = cv2.flip(display_frame, 1)
            put_meme_text(display_frame,"IMAGINATION")
            display_frame = cv2.flip(display_frame, 1)

    #print mirrored output frame
    cv2.imshow("Face + Hand Tracking", cv2.flip(display_frame, 1))
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()