import cv2
import dlib
import numpy as np
from scipy.spatial import Delaunay

# --- Path to your landmark predictor ---
path = r"shape_predictor_68_face_landmarks.dat"
if not path or not path.endswith(".dat"):
    raise FileNotFoundError("Check path")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(path)

# --- Adjustable parameters ---
scale_x = 1.05  # Width scale
scale_y = 0.95  # Height scale
square_power = 2.0  # Corner sharpness

# ---- Helper functions ----
def get_landmarks(predictor, gray, face):
    shape = predictor(gray, face)
    return np.array([(p.x, p.y) for p in shape.parts()], dtype=np.float32)

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

if __name__ == "__main__":
    # ---- Webcam ----
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    print("Press 'q' to quit")

    # 🟫 Randomly choose landmark indices for SpongeBob spots (anchored to the face)
    np.random.seed(42)
    spot_indices = np.random.choice(range(68), size=15, replace=False)
    spot_offsets = np.random.randint(-10, 10, size=(len(spot_indices), 2))
    spot_radii = np.random.randint(10, 40, size=len(spot_indices))

    # ---- Main loop ----
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = get_landmarks(predictor, gray, face)
            landmarks = add_forehead_points(landmarks, scale_y=0.7, n_points=9)

            face_center = np.mean(landmarks, axis=0)
            landmarks_target = exaggerate_points(
                landmarks, face_center,
                scale_x=scale_x, scale_y=scale_y, square_power=square_power
            )

            height, width = frame.shape[:2]
            border_pts = np.array(
                [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                dtype=np.float32
            )
            landmarks_full = np.vstack([landmarks, border_pts])
            landmarks_target_full = np.vstack([landmarks_target, border_pts])

            triangulate = Delaunay(landmarks_full)
            frame_copy = frame.copy()

            for simplex in triangulate.simplices:
                pts_src = landmarks_full[simplex]
                pts_dst = landmarks_target_full[simplex]
                warp_triangle(frame_copy, frame, pts_src, pts_dst)

            frame = cv2.bilateralFilter(frame, 9, 75, 75)

            # 🟨 --- ADD YELLOW TINT ---
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(mask, np.int32(landmarks_target), 255)
            mask = cv2.dilate(mask, np.ones((25, 25), np.uint8), iterations=1)
            mask = cv2.GaussianBlur(mask, (35, 35), 25)

            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            L, A, B = cv2.split(lab)
            B = cv2.add(B, 50)
            B = np.clip(B, 0, 255)
            lab_yellow = cv2.merge([L, A, B])
            yellow_bgr = cv2.cvtColor(lab_yellow, cv2.COLOR_LAB2BGR)

            mask_3ch = cv2.merge([mask, mask, mask])
            frame = (yellow_bgr * (mask_3ch / 255.0) +
                    frame * (1 - mask_3ch / 255.0)).astype(np.uint8)

            # 🟫 --- ADD BROWN SPOTS THAT FOLLOW LANDMARKS ---
            spot_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

            for i, idx in enumerate(spot_indices):
                cx, cy = landmarks_target[idx] + spot_offsets[i]
                radius = spot_radii[i]
                cv2.circle(spot_mask, (int(cx), int(cy)), int(radius), 255, -1)

            spot_mask = cv2.GaussianBlur(spot_mask, (41, 41), 20)

            brown_tint = frame.copy()
            brown_tint[:, :, 2] = cv2.add(brown_tint[:, :, 2], 15)   # more red
            brown_tint[:, :, 1] = cv2.subtract(brown_tint[:, :, 1], 50)  # less green
            brown_tint[:, :, 0] = cv2.subtract(brown_tint[:, :, 0], 80)  # less blue

            spot_mask_3ch = cv2.merge([spot_mask, spot_mask, spot_mask])
            frame = (brown_tint * (spot_mask_3ch / 255.0) +
                    frame * (1 - spot_mask_3ch / 255.0)).astype(np.uint8)

            break  # only first face handled

        cv2.imshow("SpongeBob Face Filter", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()