"""
utils/ear.py
============
EAR  = Eye Aspect Ratio
MAR  = Mouth Aspect Ratio

WHERE IT IS USED:
  - Called every video frame from app.py after MediaPipe returns landmarks
  - EAR feeds into: blink detection, PERCLOS calculation, CNN pre-check, ANN input
  - MAR feeds into: yawn detection, ANN input

HOW IT WORKS:
  Eye has 6 landmark points:
       p2  p3
  p1          p4
       p6  p5

  EAR = (dist(p2,p6) + dist(p3,p5)) / (2 * dist(p1,p4))

  When eye is OPEN  → EAR ≈ 0.30–0.40
  When eye is CLOSED → EAR < 0.25  ← our threshold

  Same formula adapted for mouth (MAR) with 8 mouth points.
  When mouth is OPEN (yawning) → MAR > 0.6
"""

import numpy as np


def _dist(a, b):
    """Euclidean distance between two (x, y) or (x, y, z) points."""
    return np.linalg.norm(np.array(a) - np.array(b))


def compute_ear(eye_landmarks):
    """
    Compute Eye Aspect Ratio from 6 eye landmark points.

    Args:
        eye_landmarks: list of 6 (x, y) tuples in order:
                       [p1(left), p2(top-left), p3(top-right),
                        p4(right), p5(bot-right), p6(bot-left)]

    Returns:
        float: EAR value (0.0 = closed, ~0.35 = fully open)
    """
    if len(eye_landmarks) < 6:
        return 0.3  # safe default

    p1, p2, p3, p4, p5, p6 = eye_landmarks[:6]

    # Vertical distances
    v1 = _dist(p2, p6)   # top-left to bot-left
    v2 = _dist(p3, p5)   # top-right to bot-right

    # Horizontal distance
    h = _dist(p1, p4)    # left corner to right corner

    if h < 1e-6:
        return 0.3

    ear = (v1 + v2) / (2.0 * h)
    return round(float(ear), 4)


def compute_avg_ear(left_eye_lm, right_eye_lm):
    """
    Average EAR of both eyes.

    Args:
        left_eye_lm:  list of 6 (x, y) for left eye
        right_eye_lm: list of 6 (x, y) for right eye

    Returns:
        float: average EAR
    """
    ear_l = compute_ear(left_eye_lm)
    ear_r = compute_ear(right_eye_lm)
    return round((ear_l + ear_r) / 2.0, 4)


def compute_mar(mouth_landmarks):
    """
    Compute Mouth Aspect Ratio from 8 mouth landmark points.

    Mouth points (from MediaPipe 468-pt mesh, simplified to 8):
      top-center, top-left, left-corner, bot-left,
      bot-center, bot-right, right-corner, top-right

    MAR = (vertical_distances) / (2 * horizontal_distance)
    Normal closed mouth → MAR ≈ 0.3–0.4
    Yawning             → MAR > 0.6

    Args:
        mouth_landmarks: list of 8 (x, y) tuples

    Returns:
        float: MAR value
    """
    if len(mouth_landmarks) < 8:
        return 0.3

    p1, p2, p3, p4, p5, p6, p7, p8 = mouth_landmarks[:8]

    # Vertical distances (multiple pairs for robustness)
    v1 = _dist(p2, p8)
    v2 = _dist(p3, p7)
    v3 = _dist(p4, p6)

    # Horizontal distance (left to right corner)
    h = _dist(p1, p5)

    if h < 1e-6:
        return 0.3

    mar = (v1 + v2 + v3) / (3.0 * h)
    return round(float(mar), 4)


def is_eye_closed(ear, threshold=0.25):
    """Returns True if eye is considered closed."""
    return ear < threshold


def is_yawning(mar, threshold=0.60):
    """Returns True if mouth is open wide enough to indicate yawning."""
    return mar > threshold


# ── MediaPipe index maps ──────────────────────────────────────────────────────
# MediaPipe Face Mesh uses 468 landmarks.
# These index sets map to eye / mouth regions.

# Left eye (from viewer's perspective = person's right eye)
LEFT_EYE_INDICES  = [362, 385, 387, 263, 373, 380]
# Right eye
RIGHT_EYE_INDICES = [33,  160, 158, 133, 153, 144]

# Mouth outer contour (8 key points)
MOUTH_INDICES = [61, 40, 37, 0, 267, 270, 291, 321]


def extract_eye_landmarks(face_landmarks, img_w, img_h, side='left'):
    """
    Extract normalized (x, y) pixel coords for eye landmarks from
    MediaPipe FaceMesh results.

    Args:
        face_landmarks: mediapipe face_landmarks object
        img_w, img_h:   frame dimensions
        side:           'left' or 'right'

    Returns:
        list of 6 (x_px, y_px) tuples
    """
    indices = LEFT_EYE_INDICES if side == 'left' else RIGHT_EYE_INDICES
    pts = []
    for idx in indices:
        lm = face_landmarks.landmark[idx]
        pts.append((lm.x * img_w, lm.y * img_h))
    return pts


def extract_mouth_landmarks(face_landmarks, img_w, img_h):
    """
    Extract mouth landmark pixel coords.

    Returns:
        list of 8 (x_px, y_px) tuples
    """
    pts = []
    for idx in MOUTH_INDICES:
        lm = face_landmarks.landmark[idx]
        pts.append((lm.x * img_w, lm.y * img_h))
    return pts
