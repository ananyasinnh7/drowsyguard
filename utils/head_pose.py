"""
utils/head_pose.py
==================
Head Pose Estimation using OpenCV solvePnP

WHERE IT IS USED:
  - Called every frame from app.py after MediaPipe landmarks are extracted
  - Returns pitch, yaw, roll angles
  - Yaw and Pitch are fed into the ANN as two of the 6 input features
  - High pitch (nodding) or high yaw (looking away) indicates drowsiness

HOW IT WORKS:
  We use OpenCV's solvePnP() which solves the "Perspective-n-Point" problem:
  Given N known 3D world points and their 2D projections in an image,
  find the rotation and translation of the camera (or equivalently, the object).

  Steps:
  1. Define a standard 3D face model (nose, chin, eye corners, mouth corners)
     in a canonical "model space" (millimeters, face centered at origin)
  2. Get the corresponding 2D pixel positions from MediaPipe landmarks
  3. Call cv2.solvePnP() → gives rotation vector (rvec) and translation (tvec)
  4. Convert rvec to rotation matrix using cv2.Rodrigues()
  5. Decompose rotation matrix into Euler angles (pitch, yaw, roll)

  Angle interpretation:
    pitch > 15°   → head nodding forward (drowsiness sign)
    pitch < -15°  → head tilted back
    yaw   > 20°   → looking left/right (distraction)
    roll  > 15°   → head tilting sideways

  These thresholds are used in alert.py.
"""

import numpy as np
import cv2


# ── 3D model points (generic face, in mm) ─────────────────────────────────────
# These are the approximate positions of key facial features on a
# standard human face model, used as reference for pose estimation.
MODEL_3D_POINTS = np.array([
    (0.0,    0.0,    0.0),     # Nose tip           (anchor point)
    (0.0,   -330.0, -65.0),    # Chin
    (-225.0,  170.0, -135.0),  # Left eye left corner
    (225.0,   170.0, -135.0),  # Right eye right corner
    (-150.0, -150.0, -125.0),  # Left mouth corner
    (150.0,  -150.0, -125.0),  # Right mouth corner
], dtype=np.float64)

# Corresponding MediaPipe landmark indices for each 3D model point above
LANDMARK_INDICES = [
    1,    # Nose tip
    152,  # Chin
    33,   # Left eye left corner
    263,  # Right eye right corner
    61,   # Left mouth corner
    291,  # Right mouth corner
]


def get_camera_matrix(img_w: int, img_h: int) -> np.ndarray:
    """
    Approximate camera intrinsic matrix assuming no lens distortion.
    For a real camera, you would calibrate with a checkerboard pattern.

    focal_length ≈ image_width (common approximation for webcams)

    Camera matrix K:
        [fx  0  cx]
        [0  fy  cy]
        [0   0   1]
    where cx, cy = image center, fx, fy = focal lengths
    """
    focal_length = img_w
    center = (img_w / 2.0, img_h / 2.0)
    return np.array([
        [focal_length, 0,            center[0]],
        [0,            focal_length, center[1]],
        [0,            0,            1        ]
    ], dtype=np.float64)


DIST_COEFFS = np.zeros((4, 1), dtype=np.float64)  # Assuming no lens distortion


def extract_pose_landmarks(face_landmarks, img_w: int, img_h: int):
    """
    Extract the 6 2D image points needed for solvePnP.

    Args:
        face_landmarks: MediaPipe face_landmarks result
        img_w, img_h:   frame width and height

    Returns:
        np.ndarray shape (6, 2): pixel coordinates of pose landmarks
    """
    points_2d = []
    for idx in LANDMARK_INDICES:
        lm = face_landmarks.landmark[idx]
        x = lm.x * img_w
        y = lm.y * img_h
        points_2d.append((x, y))
    return np.array(points_2d, dtype=np.float64)


def estimate_head_pose(face_landmarks, img_w: int, img_h: int):
    """
    Estimate head pose (pitch, yaw, roll) from MediaPipe landmarks.

    Args:
        face_landmarks: MediaPipe face_landmarks result
        img_w, img_h:   frame dimensions

    Returns:
        dict with keys:
            pitch (float): head nod angle in degrees (+ve = looking down)
            yaw   (float): head turn angle in degrees (+ve = looking right)
            roll  (float): head tilt angle in degrees
            deviation (float): combined angular deviation (used in ANN)
            rvec, tvec: raw rotation/translation vectors
    """
    points_2d = extract_pose_landmarks(face_landmarks, img_w, img_h)
    camera_matrix = get_camera_matrix(img_w, img_h)

    # Solve for pose
    success, rvec, tvec = cv2.solvePnP(
        MODEL_3D_POINTS,
        points_2d,
        camera_matrix,
        DIST_COEFFS,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return {"pitch": 0.0, "yaw": 0.0, "roll": 0.0, "deviation": 0.0,
                "rvec": None, "tvec": None}

    # Convert rotation vector to rotation matrix
    rmat, _ = cv2.Rodrigues(rvec)

    # Decompose rotation matrix to Euler angles
    # Using the projection matrix method
    proj_matrix = np.hstack((rmat, tvec))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)

    pitch = float(euler_angles[0])
    yaw   = float(euler_angles[1])
    roll  = float(euler_angles[2])

    # Combined deviation (magnitude, used as ANN input feature)
    deviation = float(np.sqrt(pitch**2 + yaw**2))

    return {
        "pitch":     round(pitch, 2),
        "yaw":       round(yaw, 2),
        "roll":      round(roll, 2),
        "deviation": round(min(deviation, 90.0), 2),
        "rvec":      rvec,
        "tvec":      tvec,
    }


def get_nose_direction_arrow(face_landmarks, img_w, img_h, rvec, tvec):
    """
    Compute the nose direction endpoint for drawing an arrow overlay
    showing where the head is pointing.

    Args:
        face_landmarks: MediaPipe result
        img_w, img_h:   frame dimensions
        rvec, tvec:     from estimate_head_pose()

    Returns:
        tuple: (nose_tip_2d, nose_end_2d) — both (x, y) pixel points
    """
    if rvec is None or tvec is None:
        return None, None

    camera_matrix = get_camera_matrix(img_w, img_h)

    # Project nose tip
    nose_end_3d = np.array([[0.0, 0.0, 250.0]])  # 250mm forward from nose
    nose_end_2d, _ = cv2.projectPoints(
        nose_end_3d, rvec, tvec, camera_matrix, DIST_COEFFS
    )

    nose_tip_lm = face_landmarks.landmark[1]
    nose_tip_2d = (int(nose_tip_lm.x * img_w), int(nose_tip_lm.y * img_h))
    nose_end_pt = (int(nose_end_2d[0][0][0]), int(nose_end_2d[0][0][1]))

    return nose_tip_2d, nose_end_pt
