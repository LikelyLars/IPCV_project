import cv2
import dlib
import numpy as np

# --- FACE DETECTION ---
def get_landmarks(predictor, gray, face):
    shape = predictor(gray, face)
    return np.array([(p.x, p.y) for p in shape.parts()], dtype=np.float32)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# --- WEBCAM ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# --- Trackbar callback ---
def nothing(x):
    pass

cv2.namedWindow("HSV Tuning")
cv2.createTrackbar("H Min", "HSV Tuning", 0, 179, nothing)
cv2.createTrackbar("H Max", "HSV Tuning", 20, 179, nothing)
cv2.createTrackbar("S Min", "HSV Tuning", 20, 255, nothing)
cv2.createTrackbar("S Max", "HSV Tuning", 255, 255, nothing)
cv2.createTrackbar("V Min", "HSV Tuning", 70, 255, nothing)
cv2.createTrackbar("V Max", "HSV Tuning", 255, 255, nothing)

print("Webcam opened. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    output_frame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --- FACE DETECTION ---
    faces = detector(gray)
    face_mask = np.zeros_like(gray, dtype=np.uint8)  # Mask of face regions

    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw landmarks
        landmarks = get_landmarks(predictor, gray, face)
        for (lx, ly) in landmarks:
            cv2.circle(output_frame, (int(lx), int(ly)), 2, (0, 0, 255), -1)

        # Add face region to mask
        cv2.rectangle(face_mask, (x, y), (x + w, y + h), 255, -1)

    # --- HAND DETECTION ---
    # Get HSV values from sliders
    h_min = cv2.getTrackbarPos("H Min", "HSV Tuning")
    h_max = cv2.getTrackbarPos("H Max", "HSV Tuning")
    s_min = cv2.getTrackbarPos("S Min", "HSV Tuning")
    s_max = cv2.getTrackbarPos("S Max", "HSV Tuning")
    v_min = cv2.getTrackbarPos("V Min", "HSV Tuning")
    v_max = cv2.getTrackbarPos("V Max", "HSV Tuning")

    lower_skin = np.array([h_min, s_min, v_min], dtype=np.uint8)
    upper_skin = np.array([h_max, s_max, v_max], dtype=np.uint8)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # --- IMPORTANT: Ignore face region ---
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(face_mask))

    # Clean mask
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Find contours (hands)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 2000:  # Filter out small blobs
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(output_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(output_frame, "Hand", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

    # --- DISPLAY ---
    cv2.imshow("Face + Hand Detection", output_frame)
    cv2.imshow("Hand Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
