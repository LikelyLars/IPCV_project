import cv2
import dlib
import numpy as np
import mediapipe as mp
from collections import deque
from scipy.spatial import Delaunay

# ----- initialise dlib's face detector and shape predictor ----- #
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# ----- initialise MediaPipe Hands model and other hand tracking needs ----- #
mp_hands = mp.solutions.hands
hands_model = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

tracks = [deque(maxlen=90), deque(maxlen=90)]  # left, right hand tracks

# reset hand tracks
def reset_hand_tracks():
    tracks = [deque(maxlen=90), deque(maxlen=90)]  # left, right hand tracks
    return tracks

# --- SPONGEBOB SPOT CONFIGURATION ---
# generates new spot parameters when the filter is activated
def generate_spots():
    global SPOT_INDICES, SPOT_OFFSETS, SPOT_RADII
    SPOT_INDICES = np.random.choice(range(68), size=np.random.randint(10,20), replace=False)
    SPOT_OFFSETS = np.random.randint(-10, 10, size=(len(SPOT_INDICES), 2))
    SPOT_RADII = np.random.randint(10, 20, size=len(SPOT_INDICES))


# ---- IMAGE PREPROCESSING FUNCTIONS ----
# Improve contrast and lighting balance using CLAHE(Contrast Limited Adaptive Histogram Equalization)
def adjust_lighting(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

# Auto gamma correction based on brightness
def auto_gamma_correction(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_intensity = np.mean(gray)
    gamma = np.interp(mean_intensity, [50, 150], [1.8, 0.7])
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)

# ----- FACE RECOGNITION AND LANDMARK DETECTION FUNCTIONS ----

# add additional forehead points to landmarks (extending from the eyebrows upwards)
def add_forehead_points(landmarks, scale_y=0.8, n_points=12):
    left_eyebrow = landmarks[17:22]
    right_eyebrow = landmarks[22:27]

    x_min = np.min(left_eyebrow[:, 0])
    x_max = np.max(right_eyebrow[:, 0])
    eyebrow_center_y = np.mean(np.vstack([left_eyebrow, right_eyebrow])[:, 1])

    y_offset = scale_y * np.ptp(landmarks[:, 1])

    xs = np.linspace(x_min, x_max, n_points)
    curve = []
    for i, x in enumerate(xs):
        curvature = 1 - ((i - (n_points - 1) / 2) ** 2) / ((n_points - 1) ** 2 / 4)
        y = eyebrow_center_y - y_offset * curvature
        curve.append([x, y])
    curve = np.array(curve, dtype=int)
    return np.vstack([landmarks, curve])

# returns detected faces with bounding box and landmarks
def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    detected_faces = []

    for face in faces:
        shape = predictor(gray, face)
        landmarks = np.array([(p.x, p.y) for p in shape.parts()], dtype=np.int32)
        landmarks = add_forehead_points(landmarks, scale_y=0.4, n_points=9)
        x, y, w, h = face.left(), face.top(), face.width(), face.height() #extend bbox upwards for forehead points
        detected_faces.append({"bbox": (x, y, w, h), "landmarks": landmarks})
        break # Process only the first detected face
    return detected_faces

# ------ HAND DETECTION AND TRACKING FUNCTIONS -----

# Detect hands using MediaPipe and return bounding boxes & centroids
def detect_hands(frame):
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

# save hand tracks to detect gestures
def update_hand_tracks(tracks, detected_hands):
    detected_hands = sorted(detected_hands, key=lambda h: h["center"][0])
    for i in range(min(len(detected_hands), 2)):
        tracks[i].append(detected_hands[i]["center"])
    return tracks

# Draw hand paths for testing purposes
def draw_hand_paths(tracks,frame):
    colors = [(255, 0, 0), (0, 255, 255)]  # Blue = left, Yellow = right
    for i, track in enumerate(tracks):
        for j in range(1, len(track)):
            cv2.line(frame, track[j-1], track[j], colors[i], 3)
    return frame


# ---- GESTURE DETECTION FUNCTIONS ----
# checking for rainbow gesture (hands start clos together above head, then move apart below/next to head)
def detect_rainbow_gesture(tracks, face_bbox, min_start_distance=80, end_distance_ratio=0.5):
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

# ---- SPONGEBOB FILTER FUNCTIONS ----

# Exaggerate facial landmarks to create points which to warp towards
def exaggerate_points(points, face_center, scale_x=1.6, scale_y=1.10, square_power=1.5):
    points = points.copy()
    cx, cy = face_center

    min_x = np.min(points[:, 0])
    max_x = np.max(points[:, 0])
    min_y = np.min(points[:, 1])
    max_y = np.max(points[:, 1])

    face_width = max_x - min_x
    face_height = max_y - min_y
    target_size = max(face_width, face_height) * 1.05

    for i in range(len(points)):
        dx = points[i][0] - cx
        dy = points[i][1] - cy

        nx = dx / (face_width / 2)
        ny = dy / (face_height / 2)

        if nx != 0:
            nx = abs(nx) ** (1 / square_power) * np.sign(nx)
        if ny != 0:
            ny = abs(ny) ** (1 / square_power) * np.sign(ny)

        points[i][0] = cx + nx * (target_size / 2) * scale_x
        points[i][1] = cy + ny * (target_size / 2) * scale_y

    return points

# Warp triangles based on the exagerated points generated with exaggerate_points
def warp_triangle(src, dst, tri_src, tri_dst):
    r1 = cv2.boundingRect(np.int32(tri_src))
    r2 = cv2.boundingRect(np.int32(tri_dst))
    if r1[2] == 0 or r1[3] == 0 or r2[2] == 0 or r2[3] == 0:
        return

    h, w = src.shape[:2]
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    if x1 < 0 or y1 < 0 or x1 + w1 > w or y1 + h1 > h:
        return

    t1_rect = [(p[0] - r1[0], p[1] - r1[1]) for p in tri_src]
    t2_rect = [(p[0] - r2[0], p[1] - r2[1]) for p in tri_dst]

    img1_rect = src[y1:y1 + h1, x1:x1 + w1]
    if img1_rect.size == 0:
        return

    mask = np.zeros((h2, w2, 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rect), (1.0, 1.0, 1.0), 16, 0)

    dst_crop = dst[y2:y2 + h2, x2:x2 + w2]
    if dst_crop.shape[:2] != mask.shape[:2]:
        return

    M = cv2.getAffineTransform(np.float32(t1_rect), np.float32(t2_rect))
    warped_rect = cv2.warpAffine(img1_rect, M, (w2, h2), None,
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT_101)

    dst[y2:y2 + h2, x2:x2 + w2] = dst_crop * (1 - mask) + warped_rect * mask

# combines all face warping steps, returns warped frame and new landmarks
def warp_face_region(frame, face_landmarks,SCALE_X=1.05,SCALE_Y=0.95,SQUARE_POWER=2.0):
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

# Applies spongebob like yellow tint to face region based on (exagerated) landmarks
def apply_yellow_tint(frame, landmarks_target):
    #create mask in the shape of the face
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    points = np.array(landmarks_target, dtype=np.int32)
    hull = cv2.convexHull(points) #takes outer points
    cv2.fillConvexPoly(mask, hull, 255) #creates mask inside hull

    # Convert to LAB and shift toward yellow
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    B = cv2.add(B, 100)
    B = np.clip(B, 0, 255)
    lab_yellow = cv2.merge([L, A, B])
    yellow_bgr = cv2.cvtColor(lab_yellow, cv2.COLOR_LAB2BGR)

    mask_3ch = cv2.merge([mask, mask, mask])
    blended = (yellow_bgr * (mask_3ch / 255.0) + frame * (1 - mask_3ch / 255.0)).astype(np.uint8)

    return blended

# Applies brown spots to face region. Spots are randomly generated when filter is activated
def apply_brown_tint(frame, landmarks_target, lab_shift=(20, 15), delta_L=-60):
    """
    Draw darker brown spots in LAB color space that follow facial landmarks.
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = cv2.split(lab)

    # --- 1. Create spot mask ---
    spot_mask = np.zeros(frame.shape[:2], dtype=np.float32)

    for i, idx in enumerate(SPOT_INDICES):
        cx, cy = landmarks_target[idx] + SPOT_OFFSETS[i]
        cv2.circle(
            spot_mask,
            (int(cx), int(cy)),
            int(SPOT_RADII[i]),
            1.0,
            -1
        )

    # Smooth edges of spots
    spot_mask = cv2.GaussianBlur(spot_mask, (15, 15), 5)
    spot_mask = np.clip(spot_mask, 0, 1)

    #ensure spots stay within face region
    face_mask = np.zeros(frame.shape[:2], dtype=np.float32)
    points = np.array(landmarks_target, dtype=np.int32)
    hull = cv2.convexHull(points)
    cv2.fillConvexPoly(face_mask, hull, 1.0)
    spot_mask *= face_mask  # keep spots inside face

    # apply spot mask to LAB channels
    L_spot = L + delta_L * spot_mask           # darken (delta_L < 0)
    A_spot = A + lab_shift[0] * spot_mask      # shift toward red
    B_spot = B + lab_shift[1] * spot_mask      # shift toward yellow/brown

    #merge channels back
    lab_brown = cv2.merge([L_spot, A_spot, B_spot])
    lab_brown = np.clip(lab_brown, 0, 255).astype(np.uint8)

    #convert back to BGR
    brown_bgr = cv2.cvtColor(lab_brown, cv2.COLOR_LAB2BGR)

    return brown_bgr

import cv2

# add meme-style text at the bottom of the frame
def put_meme_text(frame, text, font_scale=2, thickness=4, bottom_margin=20):
    font = cv2.FONT_HERSHEY_DUPLEX
    outline_thickness = thickness + 2

    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # Calculate bottom-center position
    x = (frame.shape[1] - text_width) // 2
    y = frame.shape[0] - bottom_margin

    # Draw black outline by drawing text multiple times offset
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx != 0 or dy != 0:
                cv2.putText(
                    frame, text, (x + dx, y + dy),
                    font, font_scale, (0, 0, 0),
                    thickness=outline_thickness,
                    lineType=cv2.LINE_AA
                )

    # Draw white text on top
    cv2.putText(
        frame, text, (x, y),
        font, font_scale, (255, 255, 255),
        thickness=thickness,
        lineType=cv2.LINE_AA
    )

    return frame

#overlay image with alpha channel handling (when the rainbow is not fully transparent, show it)
def overlay_image_alpha(bg, fg, x, y):
    fg_h, fg_w = fg.shape[:2]

    # Clip the overlay region to fit inside background
    if x < 0:
        fg = fg[:, -x:]
        fg_w += x
        x = 0
    if y < 0:
        fg = fg[-y:, :]
        fg_h += y
        y = 0
    if x + fg_w > bg.shape[1]:
        fg = fg[:, :bg.shape[1]-x]
        fg_w = fg.shape[1]
    if y + fg_h > bg.shape[0]:
        fg = fg[:bg.shape[0]-y, :]
        fg_h = fg.shape[0]

    # Separate color and alpha channels
    if fg.shape[2] == 4:
        alpha = fg[:, :, 3] / 255.0
        fg_rgb = fg[:, :, :3]
    else:
        alpha = np.ones((fg_h, fg_w))
        fg_rgb = fg

    # Overlay
    bg_slice = bg[y:y+fg_h, x:x+fg_w]
    for c in range(3):
        bg_slice[:, :, c] = fg_rgb[:, :, c] * alpha + bg_slice[:, :, c] * (1-alpha)

    bg[y:y+fg_h, x:x+fg_w] = bg_slice
    return bg




