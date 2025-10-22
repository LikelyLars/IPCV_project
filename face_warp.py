import cv2
import dlib
import numpy as np
from scipy.spatial import Delaunay

# TODO: Path aanpassen voor jezelf.
path = r"D:/Documenten/Python/BMT_Q1/IP_CV/Project_persoonlijk/shape_predictor_68_face_landmarks.dat"

predictor_path = path  # replace with your path
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)


# ---- Helper functions ----
def get_landmarks(predictor, gray, face):
    shape = predictor(gray, face)
    return np.array([(p.x, p.y) for p in shape.parts()], dtype=np.float32)


def add_forehead_points(landmarks, scale_y=0.5, n_points=9):
    """
    Add a smooth forehead curve above the eyebrows and into the hairline.
    n_points controls the smoothness (7–11 works well).
    """
    left_eyebrow = landmarks[17:22]
    right_eyebrow = landmarks[22:27]

    # Horizontal range and eyebrow center
    x_min = np.min(left_eyebrow[:, 0])
    x_max = np.max(right_eyebrow[:, 0])
    eyebrow_center_y = np.mean(np.vstack([left_eyebrow, right_eyebrow])[:, 1])

    # How high to go above eyebrows
    y_offset = scale_y * np.ptp(landmarks[:, 1])

    # Generate a smooth parabola-like forehead curve
    xs = np.linspace(x_min, x_max, n_points)
    curve = []
    for i, x in enumerate(xs):
        curvature = 1 - ((i - (n_points - 1) / 2) ** 2) / ((n_points - 1) ** 2 / 4)
        y = eyebrow_center_y - y_offset * curvature
        curve.append([x, y])
    curve = np.array(curve, dtype=np.float32)

    return np.vstack([landmarks, curve])

def exaggerate_points(points, face_center, scale_x=1.45, scale_y=1.10, square_power=1.25):
    """
    Exaggerate face shape into a more square, inflated form.
    - scale_x: horizontal stretch (more = wider)
    - scale_y: vertical stretch (less = flatter)
    - square_power: nonlinear squaring factor to flatten sides
    """
    pts = points.copy()
    cx, cy = face_center

    # Compute the maximum horizontal distance (for corner_factor scaling)
    max_x_dist = max(abs(p[0] - cx) for p in pts)

    for i in range(len(pts)):
        vec = pts[i] - face_center
        dist = np.linalg.norm(vec)
        if dist == 0:
            continue  # skip the center point to avoid division by zero
        nx, ny = vec / dist  # normalized direction vector

        # Nonlinear scaling: sides pop more than center
        fx = (abs(nx) ** square_power) * np.sign(nx)
        fy = (abs(ny) ** (square_power * 0.7)) * np.sign(ny)  # less aggressive vertically

        # Corner boost: amplify stretch near horizontal edges
        dist_x = pts[i][0] - cx
        corner_factor = 1 + 0.25 * (abs(dist_x) / max_x_dist)  # 0.25 = more square, tune as needed

        # Apply exaggerated scaling
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
        landmarks = add_forehead_points(landmarks, scale_y=0.8, n_points=9)
        face_center = np.mean(landmarks, axis=0)

        landmarks_target = exaggerate_points(landmarks, face_center, scale_x=1.3, scale_y=1.15)

        # Add border points so the warp covers the whole head (and hair)
        h, w = frame.shape[:2]
        border_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
        landmarks_full = np.vstack([landmarks, border_pts])
        landmarks_target_full = np.vstack([landmarks_target, border_pts])

        # Triangulate
        tri = Delaunay(landmarks_full)
        frame_copy = frame.copy()

        # Warp each triangle
        for simplex in tri.simplices:
            pts_src = landmarks_full[simplex]
            pts_dst = landmarks_target_full[simplex]
            warp_triangle(frame_copy, frame, pts_src, pts_dst)

        # Optional: apply slight smoothing to blend seams
        frame = cv2.bilateralFilter(frame, 9, 75, 75)

        break  # only first detected face

    cv2.imshow("Rounded Square Snapchat-like Face", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

#| Parameter          | Description                         | Suggested Value |
#| ------------------ | ----------------------------------- | --------------- |
#| scale_x            | Horizontal stretch (wider)          | 1.45 – 1.6      |
#| scale_y            | Vertical stretch (slightly flatter) | 1.05 – 1.15     |
#| square_power       | Nonlinear squaring                  | 1.2 – 1.35      |
#| scale_y (forehead) | Raise forehead points               | 0.5 – 0.6       |  


