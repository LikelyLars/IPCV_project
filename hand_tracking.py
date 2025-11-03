import cv2
import dlib
import numpy as np
import mediapipe as mp
from collections import deque
from face_augment_2 import add_forehead_points, exaggerate_points, warp_triangle
from scipy.spatial import Delaunay

# --- INITIALIZATION ---
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

mp_hands = mp.solutions.hands
hands_model = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

tracks = [deque(maxlen=90), deque(maxlen=90)]  # left, right hand tracks

# --- SPONGEBOB SPOT CONFIGURATION ---
np.random.seed(42)
SPOT_INDICES = np.random.choice(range(68), size=15, replace=False)
SPOT_OFFSETS = np.random.randint(-10, 10, size=(len(SPOT_INDICES), 2))
SPOT_RADII = np.random.randint(10, 40, size=len(SPOT_INDICES))

# --- SCALING PARAMETERS ---
SCALE_X = 1.05
SCALE_Y = 0.95
SQUARE_POWER = 2.0

# --- IMAGE ENHANCEMENT HELPERS ---
def adjust_lighting(image):
    """Improve contrast and lighting balance using CLAHE."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def auto_gamma_correction(image):
    """Auto gamma correction based on brightness."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_intensity = np.mean(gray)
    gamma = np.interp(mean_intensity, [50, 150], [1.8, 0.7])
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)

# --- FACE DETECTION ---
def detect_faces(frame, detector, predictor):
    """Detect faces and their landmarks."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    detected_faces = []

    for face in faces:
        shape = predictor(gray, face)
        landmarks = np.array([(p.x, p.y) for p in shape.parts()], dtype=np.int32)
        landmarks = add_forehead_points(landmarks, scale_y=0.4, n_points=9)
        x, y, w, h = face.left(), face.top() - 50, face.width(), face.height() + 100
        detected_faces.append({"bbox": (x, y, w, h), "landmarks": landmarks})
    return detected_faces

# --- HAND DETECTION ---
def detect_hands(frame, hands_model):
    """Detect hands using MediaPipe and return bounding boxes + centroids."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_model.process(rgb)
    detected_hands = []

    if results.multi_hand_landmarks:
        h, w, _ = frame.shape
        for hand_landmarks in results.multi_hand_landmarks:
            x_coords = [lm.x * w for lm in hand_landmarks.landmark]
            y_coords = [lm.y * h for lm in hand_landmarks.landmark]
            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))
            cx, cy = int(np.mean(x_coords)), int(np.mean(y_coords))
            detected_hands.append({
                "bbox": (x_min, y_min, x_max - x_min, y_max - y_min),
                "center": (cx, cy)
            })
    return detected_hands

# --- HAND TRACKING ---
def update_hand_tracks(tracks, detected_hands):
    """Maintain separate tracks for up to 2 hands."""
    detected_hands = sorted(detected_hands, key=lambda h: h["center"][0])
    for i in range(min(len(detected_hands), 2)):
        tracks[i].append(detected_hands[i]["center"])
    return tracks

# def draw_hand_paths(frame, tracks):
#     """Draw hand motion trails for debugging."""
#     colors = [(255, 0, 0), (0, 255, 255)]  # Blue = left, Yellow = right
#     for i, track in enumerate(tracks):
#         for j in range(1, len(track)):
#             cv2.line(frame, track[j-1], track[j], colors[i], 3)
#     return frame

# --- GESTURE DETECTION ---
def detect_rainbow_gesture(tracks, face_bbox, min_start_distance=60, end_distance_ratio=0.5):
    """Detect a rainbow-like gesture over the head."""
    if len(tracks) < 2 or len(tracks[0]) < 5 or len(tracks[1]) < 5:
        return False

    head_x, head_y, head_w, head_h = face_bbox
    head_top = head_y

    start_idx = None
    for i in range(min(len(tracks[0]), len(tracks[1]))):
        h1, h2 = tracks[0][i], tracks[1][i]
        hands_close = np.linalg.norm(np.array(h1) - np.array(h2)) < min_start_distance
        above_head = h1[1] < head_top and h2[1] < head_top
        if hands_close and above_head:
            start_idx = i
            break

    if start_idx is None:
        return False

    h1_end, h2_end = tracks[0][-1], tracks[1][-1]
    hands_apart = (h1_end[0] < head_x + head_w * end_distance_ratio and
                   h2_end[0] > head_x + head_w * (1 - end_distance_ratio)) or \
                  (h2_end[0] < head_x + head_w * end_distance_ratio and
                   h1_end[0] > head_x + head_w * (1 - end_distance_ratio))
    hands_below = h1_end[1] >= head_top and h2_end[1] >= head_top

    return hands_apart and hands_below

def warp_face_region(frame, face_landmarks):
    """Apply geometric warping to exaggerate facial features."""
    height, width = frame.shape[:2]
    face_center = np.mean(face_landmarks, axis=0)

    # Exaggerate points (custom function from your module)
    landmarks_target = exaggerate_points(
        face_landmarks, face_center,
        scale_x=SCALE_X, scale_y=SCALE_Y, square_power=SQUARE_POWER
    )

    # Add frame corners to avoid edge distortion
    border_pts = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32
    )

    landmarks_full = np.vstack([face_landmarks, border_pts])
    landmarks_target_full = np.vstack([landmarks_target, border_pts])

    triangulation = Delaunay(landmarks_full)
    warped = frame.copy()

    for simplex in triangulation.simplices:
        pts_src = landmarks_full[simplex]
        pts_dst = landmarks_target_full[simplex]
        warp_triangle(frame, warped, pts_src, pts_dst)

    # Smooth the final warped image slightly
    warped = cv2.bilateralFilter(warped, 9, 75, 75)

    return warped, landmarks_target

def apply_yellow_tint(frame, landmarks_target):
    """Apply a soft yellow tint inside the warped face region."""
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(landmarks_target), 255)

    # Expand + soften the mask
    mask = cv2.dilate(mask, np.ones((25, 25), np.uint8), iterations=1)
    mask = cv2.GaussianBlur(mask, (35, 35), 25)

    # Convert to LAB and shift toward yellow
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    B = cv2.add(B, 50)
    B = np.clip(B, 0, 255)
    lab_yellow = cv2.merge([L, A, B])
    yellow_bgr = cv2.cvtColor(lab_yellow, cv2.COLOR_LAB2BGR)

    mask_3ch = cv2.merge([mask, mask, mask])
    blended = (yellow_bgr * (mask_3ch / 255.0) +
               frame * (1 - mask_3ch / 255.0)).astype(np.uint8)

    return blended

def apply_brown_spots(frame, landmarks_target, spot_indices, spot_offsets, spot_radii):
    """Draw brown spots that follow facial landmarks."""
    spot_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    for i, idx in enumerate(spot_indices):
        cx, cy = landmarks_target[idx] + spot_offsets[i]
        cv2.circle(spot_mask, (int(cx), int(cy)), int(spot_radii[i]), 255, -1)

    spot_mask = cv2.GaussianBlur(spot_mask, (41, 41), 20)

    # Create brown tint
    brown_tinted = frame.copy()
    brown_tinted[:, :, 2] = cv2.add(brown_tinted[:, :, 2], 15)   # More red
    brown_tinted[:, :, 1] = cv2.subtract(brown_tinted[:, :, 1], 50)  # Less green
    brown_tinted[:, :, 0] = cv2.subtract(brown_tinted[:, :, 0], 80)  # Less blue

    spot_mask_3ch = cv2.merge([spot_mask, spot_mask, spot_mask])
    combined = (brown_tinted * (spot_mask_3ch / 255.0) +
                frame * (1 - spot_mask_3ch / 255.0)).astype(np.uint8)

    return combined

# --- MAIN LOOP ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print(" Error: Could not open webcam.")
    exit()

print("Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    display_frame = frame.copy()
    processed_frame = auto_gamma_correction(adjust_lighting(frame))

    faces = detect_faces(processed_frame, detector, predictor)
    for face_data in faces:
        x, y, w, h = face_data["bbox"]
        landmarks = face_data["landmarks"]

        # cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # for (lx, ly) in landmarks:
        #     cv2.circle(display_frame, (int(lx), int(ly)), 2, (0, 0, 255), -1)

    detected_hands = detect_hands(processed_frame, hands_model)
    if detected_hands:
        tracks = update_hand_tracks(tracks, detected_hands)
        # for i, hand in enumerate(detected_hands):
        #     x, y, w, h = hand["bbox"]
        #     cx, cy = hand["center"]
        #     cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        #     cv2.circle(display_frame, (cx, cy), 6, (0, 0, 255), -1)
        #     cv2.putText(display_frame, f"Hand {i + 1}", (x, y - 10),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    if faces:
        face_data = faces[0]
        face_bbox = face_data["bbox"]
        if detect_rainbow_gesture(tracks, face_bbox):
            # cv2.putText(display_frame, "Rainbow Gesture Detected!", (50, 50),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            
            # --- Warp the face ---
            warped_frame, landmarks_target = warp_face_region(frame, face_data["landmarks"])

            # --- Apply yellow tint ---
            yellowed_frame = apply_yellow_tint(warped_frame, landmarks_target)

            # --- Add brown spots ---
            final_frame = apply_brown_spots(
                yellowed_frame, landmarks_target,
                SPOT_INDICES, SPOT_OFFSETS, SPOT_RADII
            )

            display_frame = final_frame

    #display_frame = draw_hand_paths(display_frame, tracks) #only for testing purposes
    cv2.imshow("Face + Hand Tracking", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
