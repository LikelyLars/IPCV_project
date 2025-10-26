import cv2
import dlib
import numpy as np
from scipy.spatial import Delaunay

# TODO: Path aanpassen voor jezelf.
path = r"shape_predictor_68_face_landmarks.dat"
if not path or not path.endswith(".dat"):
    raise FileNotFoundError("Check path")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(path)

# Start with adjustable parameters
scale_x = 1.05  # Overall width scale
scale_y = 0.95  # Overall height scale
square_power = 2.0  # Sharpness of corners

# Helper functions
def get_landmarks(predictor, gray, face):
    shape = predictor(gray, face)
    return np.array([(p.x, p.y) for p in shape.parts()], dtype=np.float32)

def add_forehead_points(landmarks, scale_y=0.8, n_points=12):
    # TODO: Add points above eyebrows for hariline
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
    curve = np.array(curve, dtype=np.float32)
    return np.vstack([landmarks, curve])

def exaggerate_points(points, face_center, scale_x=1.6, scale_y=1.10, square_power=1.5):
    # TODO: Change values scale_x, scale_y, square_power -> see values at the end of the file
    points = points.copy()
    cx, cy = face_center
    
    min_x = np.min(points[:, 0])
    max_x = np.max(points[:, 0])
    min_y = np.min(points[:, 1])
    max_y = np.max(points[:, 1])
    
    face_width = max_x - min_x
    face_height = max_y - min_y
    
    target_size = max(face_width, face_height) * 1.05 # 1.05 or 1.1 work best
    
    for i in range(len(points)):
        dx = points[i][0] - cx
        dy = points[i][1] - cy
        
        nx = dx / (face_width / 2)
        ny = dy / (face_height / 2)
        
        if nx != 0:
            nx = abs(nx) ** (1/square_power) * np.sign(nx)
        if ny != 0:
            ny = abs(ny) ** (1/square_power) * np.sign(ny)
        
        points[i][0] = cx + nx * (target_size/2) * scale_x
        points[i][1] = cy + ny * (target_size/2) * scale_y
        
    return points

def warp_triangle(src, dst, tri_src, tri_dst): # warping using affine transform
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
        landmarks = add_forehead_points(landmarks, scale_y=0.7, n_points=9)

        face_center = np.mean(landmarks, axis=0)
        landmarks_target = exaggerate_points(landmarks, face_center, 
                                          scale_x=scale_x, 
                                          scale_y=scale_y, 
                                          square_power=square_power)
        
        height, width = frame.shape[:2] # Also include forehead and hair
        border_pts = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
        landmarks_full = np.vstack([landmarks, border_pts])
        landmarks_target_full = np.vstack([landmarks_target, border_pts])

        triangulate = Delaunay(landmarks_full)
        frame_copy = frame.copy()

        for simplex in triangulate.simplices: 
            pts_src = landmarks_full[simplex]
            pts_dst = landmarks_target_full[simplex]
            warp_triangle(frame_copy, frame, pts_src, pts_dst)

        frame = cv2.bilateralFilter(frame, 9, 75, 75)

        break 

    # Use for parameter control display
    #param_display = f"Width: {scale_x:.2f} | Height: {scale_y:.2f} | Square: {square_power:.2f}"
    #cv2.putText(frame, param_display, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow("Square Face Filter", frame)
    key = cv2.waitKey(1) & 0xFF
    
    # Parameter controls
    # TODO: Gebruiken voor vinden goede waarden scale_x, scale_y, square_power -> Staan nu aan het begin van dit bestand
    if key == ord('q'):
        break
#    elif key == ord('w'):  # Increase width
#        scale_x += 0.05
#    elif key == ord('s'):  # Decrease width
#        scale_x -= 0.05
#    elif key == ord('e'):  # Increase height
#        scale_y += 0.05
#    elif key == ord('d'):  # Decrease height
#        scale_y -= 0.05
#    elif key == ord('r'):  # More square corners
#        square_power -= 0.1  # Lower value = more square
#    elif key == ord('f'):  # More rounded corners
#        square_power += 0.1  # Higher value = more rounded

cap.release()
cv2.destroyAllWindows()
