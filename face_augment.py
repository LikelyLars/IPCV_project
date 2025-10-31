import cv2
import dlib
import numpy as np
from scipy.spatial import Delaunay

# --- Path to your landmark predictor ---
path = r"shape_predictor_68_face_landmarks.dat"
predictor_path = path
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)


# ---- Helper functions ----
def get_landmarks(predictor, gray, face):
    shape = predictor(gray, face)
    return np.array([(p.x, p.y) for p in shape.parts()], dtype=np.float32)


def exaggerate_points(points, face_center, scale_x=1.45, scale_y=1.10, square_power=1.25):
    pts = points.copy()
    cx, cy = face_center

    max_x_dist = max(abs(p[0] - cx) for p in pts)

    for i in range(len(pts)):
        vec = pts[i] - face_center
        dist = np.linalg.norm(vec)
        if dist == 0:
            continue
        nx, ny = vec / dist

        fx = (abs(nx) ** square_power) * np.sign(nx)
        fy = (abs(ny) ** (square_power * 0.7)) * np.sign(ny)
        dist_x = pts[i][0] - cx
        corner_factor = 1 + 0.25 * (abs(dist_x) / max_x_dist)

        pts[i][0] = cx + dist * fx * scale_x * corner_factor
        pts[i][1] = cy + dist * fy * scale_y

    return pts


def warp_triangle(src, dst, tri_src, tri_dst):
    r1 = cv2.boundingRect(np.int32(tri_src))
    r2 = cv2.boundingRect(np.int32(tri_dst))
    if r1[2] == 0 or r1[3] == 0 or r2[2] == 0 or r2[3] == 0:
        return

    t1_rect = [(p[0] - r1[0], p[1] - r1[1]) for p in tri_src]
    t2_rect = [(p[0] - r2[0], p[1] - r2[1]) for p in tri_dst]

    img1_rect = src[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    warp_mat = cv2.getAffineTransform(np.float32(t1_rect), np.float32(t2_rect))
    warped_rect = cv2.warpAffine(img1_rect, warp_mat, (r2[2], r2[3]),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rect), (1.0, 1.0, 1.0), 16, 0)

    dst_crop = dst[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]
    if dst_crop.shape[:2] != mask.shape[:2]:
        return
    dst[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = dst_crop * (1 - mask) + warped_rect * mask


# ---- Webcam ----
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

print("Press 'q' to quit")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = get_landmarks(predictor, gray, face)
        face_center = np.mean(landmarks, axis=0)

        landmarks_target = exaggerate_points(landmarks, face_center, scale_x=1.3, scale_y=1.15)

        # Add border points so the warp covers the full face area
        h, w = frame.shape[:2]
        border_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
        landmarks_full = np.vstack([landmarks, border_pts])
        landmarks_target_full = np.vstack([landmarks_target, border_pts])

        # Triangulate and warp
        tri = Delaunay(landmarks_full)
        frame_copy = frame.copy()

        for simplex in tri.simplices:
            pts_src = landmarks_full[simplex]
            pts_dst = landmarks_target_full[simplex]
            warp_triangle(frame_copy, frame, pts_src, pts_dst)

        # Optional smoothing for seams
        frame = cv2.bilateralFilter(frame, 9, 75, 75)

        # 🟨 --- ADD YELLOW TINT BASED ON FACE POLYGON ---
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.int32(landmarks_target), 255)

        # Slight expansion and soft blending
        mask = cv2.dilate(mask, np.ones((25, 25), np.uint8), iterations=1)
        mask = cv2.GaussianBlur(mask, (35, 35), 25)

        # LAB-based yellow tint
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        B = cv2.add(B, 35)  # increase yellow
        B = np.clip(B, 0, 255)
        lab_yellow = cv2.merge([L, A, B])
        yellow_bgr = cv2.cvtColor(lab_yellow, cv2.COLOR_LAB2BGR)

        # Blend yellow tint only where mask applies
        mask_3ch = cv2.merge([mask, mask, mask])
        frame = (yellow_bgr * (mask_3ch / 255.0) + frame * (1 - mask_3ch / 255.0)).astype(np.uint8)

        break  # only handle the first detected face

    cv2.imshow("SpongeBob Face", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
