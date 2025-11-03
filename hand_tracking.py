import cv2
import dlib
import numpy as np
import mediapipe as mp
from collections import deque
from face_augment_2 import add_forehead_points, exaggerate_points, warp_triangle

# --- INITIALIZE STUFF ---
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

mp_hands = mp.solutions.hands
hands_model = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

tracks = [deque(maxlen=90), deque(maxlen=90)]  # two tracks: left, right

# --- helper functions for image detection ---
def adjust_lighting(image):
    """Improve contrast and lightnig balance using CLAHE."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def auto_gamma_correction(image):
    """Automatically correct gamma based on average brightness."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_intensity = np.mean(gray)
    gamma = np.interp(mean_intensity, [50, 150], [1.8, 0.7])
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)

# --- FACE DETECTION ---
#detect faces, used to know the relation between hands and head
def detect_faces(frame, detector, predictor):
    """
    Detects faces and their landmarks.
    Returns: list of dicts with 'bbox' and 'landmarks'.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    detected_faces = []

    for face in faces:  
        shape = predictor(gray, face)
        landmarks = np.array([(p.x, p.y) for p in shape.parts()], dtype=np.int32)
        landmarks = add_forehead_points(landmarks, scale_y=0.4, n_points=9)
        x, y, w, h = face.left(), face.top()-50, face.width(), face.height()+100 #make the box bigger to include forehead
        detected_faces.append({"bbox": (x, y, w, h), "landmarks": landmarks})
    return detected_faces

# --- HAND DETECTION + TRACKING ---
#detect hands, to track their their positions
def detect_hands(frame, hands_model):
    """Detect hands using MediaPipe and return bounding boxes + centroids."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands_model.process(rgb)
    detected = []
    if result.multi_hand_landmarks:
        h, w, _ = frame.shape
        for hand_landmarks in result.multi_hand_landmarks:
            x_coords = [lm.x * w for lm in hand_landmarks.landmark]
            y_coords = [lm.y * h for lm in hand_landmarks.landmark]
            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))
            cx, cy = int(np.mean(x_coords)), int(np.mean(y_coords))
            detected.append({"bbox": (x_min, y_min, x_max - x_min, y_max - y_min), "center": (cx, cy)})
    return detected

#track motion of the hands
def update_hand_tracks(tracks, detected_hands, max_len=50):
    """Maintain separate tracks for up to 2 hands."""
    detected_hands = sorted(detected_hands, key=lambda h: h["center"][0])
    if len(detected_hands) == 2:
        for i, hand in enumerate(detected_hands):
            tracks[i].append(hand["center"])
    return tracks

#dra the paths, for testing
def draw_hand_paths(frame, tracks):
    """Draw each hand’s movement path."""
    colors = [(255, 0, 0), (0, 255, 255)]  # Blue = left, Yellow = right
    for i, track in enumerate(tracks):
        for j in range(1, len(track)):
            cv2.line(frame, track[j-1], track[j], colors[i], 3)
    return frame

# --- GESTURE DETECTION ---
#detect the rainbow gesture of the hands aove the head, and returns true if detected
def detect_rainbow_gesture(tracks, face_bbox, min_start_distance=50, min_end_distance_ratio=0.5):
    """
    Detects if hands performed a "rainbow over the head" gesture dynamically.
    """
    if len(tracks) != 2 or len(tracks[0]) < 5 or len(tracks[1]) < 5:
        return False  # Not enough data yet

    head_x, head_y, head_w, head_h = face_bbox
    head_top_y = head_y

    # --- Find the first frame where hands are close together above the head ---
    start_idx = None
    for i in range(len(tracks[0])):
        h1 = tracks[0][i]
        h2 = tracks[1][i]
        hands_close = np.linalg.norm(np.array(h1) - np.array(h2)) < min_start_distance
        hands_above_head = h1[1] < head_top_y and h2[1] < head_top_y
        if hands_close and hands_above_head:
            start_idx = i
            break

    if start_idx is None:
        return False  # No valid start found yet, so no gesture done

    # --- Use last frame as end position ---
    hand1_end = tracks[0][-1]
    hand2_end = tracks[1][-1]

    # --- End positions on sides of head ---
    hands_apart = (hand1_end[0] < head_x + head_w * min_end_distance_ratio and
                   hand2_end[0] > head_x + head_w * (1 - min_end_distance_ratio)) or \
                  (hand2_end[0] < head_x + head_w * min_end_distance_ratio and #doesnt matter which hand goed to which side
                   hand1_end[0] > head_x + head_w * (1 - min_end_distance_ratio))

    hands_below_head = hand1_end[1] >= head_top_y and hand2_end[1] >= head_top_y

    return hands_apart and hands_below_head

# --- MAIN LOOP ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    #make the frame more robust, but output the original
    output_frame = frame.copy()
    processed_frame = auto_gamma_correction(adjust_lighting(frame))

    # --- FACE DETECTION ---
    faces = detect_faces(processed_frame, detector, predictor)
    for face in faces:
        x, y, w, h = face["bbox"]
        cv2.rectangle(output_frame, (x, y), (x+w, y+h), (0, 255, 0), 2) #draw box and landmarks for testing
        for (lx, ly) in face["landmarks"]:
            cv2.circle(output_frame, (lx, ly), 2, (0, 0, 255), -1)

    # --- HAND DETECTION + TRACKING ---
    detected_hands = detect_hands(processed_frame, hands_model)
    if len(detected_hands) == 2:
        tracks = update_hand_tracks(tracks, detected_hands)
        for i, hand in enumerate(detected_hands):
            x, y, w, h = hand["bbox"]
            cx, cy = hand["center"]
            cv2.rectangle(output_frame, (x, y), (x+w, y+h), (255, 0, 0), 2) #draw for testing
            cv2.circle(output_frame, (cx, cy), 6, (0, 0, 255), -1)
            cv2.putText(output_frame, f"Hand {i+1}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
            
    # --- GESTURE DETECTION ---
    if faces:  # assume single face
        face_bbox = faces[0]["bbox"]
        rainbow_done = detect_rainbow_gesture(tracks, face_bbox)
        if rainbow_done:
            cv2.putText(output_frame, "Rainbow Gesture Detected!", (50,50), #say gesture detected for testing
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 3)

    # --- DRAW TRAILS (only for testing) ---
    output_frame = draw_hand_paths(output_frame, tracks)

    # --- DISPLAY ---
    cv2.imshow("Face + Hand Tracking", output_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()