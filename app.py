"""
app.py
======
DrowsyGuard — Flask Backend Server

WHERE THIS FILE SITS IN THE SYSTEM:
  This is the ENTRY POINT. It:
  1. Serves the HTML frontend at http://localhost:5000
  2. Opens the webcam using OpenCV
  3. Runs MediaPipe face detection every frame
  4. Calls EAR, PERCLOS, HeadPose, CNN, ANN utilities
  5. Sends processed metrics to the frontend via HTTP polling (/api/metrics)
  6. Handles all button actions via REST API endpoints

PROCESSING PIPELINE PER FRAME:
  Camera Frame (BGR)
       ↓
  MediaPipe FaceMesh  ← detects 468 face landmarks
       ↓
  ┌─────────────────────────────────────────────────┐
  │  utils/ear.py      → EAR (left eye, right eye)  │
  │  utils/ear.py      → MAR (mouth aspect ratio)   │
  │  utils/perclos.py  → PERCLOS (60s window)       │
  │  utils/perclos.py  → blink detection + rate     │
  │  utils/head_pose.py→ pitch, yaw, roll angles    │
  │  models/cnn_model.py→ CNN eye closed prob       │ ← all run in parallel
  └─────────────────────────────────────────────────┘
       ↓
  models/ann_model.py  → ANN fatigue score (0–100)
       ↓
  utils/alert.py       → alert level (0, 1, 2, 3)
       ↓
  JSON response → frontend every ~100ms

HOW TO RUN:
  python app.py
  Then open: http://localhost:5000
"""

import cv2
import numpy as np
import base64
import time
import json
import threading
import os
from flask import Flask, render_template, jsonify, request, Response

# Try importing flask_cors (graceful fallback if not installed)
try:
    from flask_cors import CORS
    CORS_AVAILABLE = True
except ImportError:
    CORS_AVAILABLE = False

# Try importing mediapipe
try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False
    print("[WARNING] mediapipe not installed. Run: pip install mediapipe")

# Our custom modules
from utils.ear import (compute_avg_ear, compute_mar,
                        extract_eye_landmarks, extract_mouth_landmarks,
                        is_eye_closed, is_yawning)
from utils.perclos import PERCLOSTracker, BlinkTracker
from utils.head_pose import estimate_head_pose, get_nose_direction_arrow
from utils.alert import AlertManager
from models.cnn_model import EyeCNN, crop_eye_region
from models.ann_model import FatigueANN

# ──────────────────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder='templates', static_folder='static')
if CORS_AVAILABLE:
    CORS(app)

os.makedirs('static/snapshots', exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
#  GLOBAL STATE
#  All state is kept here and protected by a threading lock
# ──────────────────────────────────────────────────────────────────────────────
state = {
    "running":         False,
    "paused":          False,
    "cap":             None,    # OpenCV VideoCapture object
    "thread":          None,    # Background detection thread

    # Raw metrics (updated every frame)
    "ear":             0.0,
    "mar":             0.0,
    "perclos":         0.0,
    "blink_rate":      0,
    "head_pitch":      0.0,
    "head_yaw":        0.0,
    "head_roll":       0.0,
    "cnn_prob":        0.0,     # CNN: P(eye closed)
    "fatigue_score":   0.0,     # ANN output 0–100
    "alert_level":     0,       # 0–3
    "alert_name":      "System Ready",
    "alert_desc":      "Click Start Camera to begin",
    "alert_color":     "#00C68A",
    "face_detected":   False,

    # Session
    "session_start":   None,
    "total_alerts":    0,
    "total_blinks":    0,
    "fps":             0,

    # Feature breakdown (from ANN)
    "breakdown":       {},

    # Settings (can be changed via API)
    "ear_threshold":   0.25,
    "show_overlay":    True,
    "sensitivity":     2,       # 1=low, 2=medium, 3=high

    # Current frame as JPEG base64 (for /api/frame endpoint)
    "frame_b64":       None,
}

state_lock = threading.Lock()

# ──────────────────────────────────────────────────────────────────────────────
#  AI COMPONENTS — instantiated once at startup
# ──────────────────────────────────────────────────────────────────────────────
perclos_tracker = PERCLOSTracker(window_sec=60)
blink_tracker   = BlinkTracker()
alert_manager   = AlertManager(hysteresis_frames=15)
cnn             = EyeCNN()
ann             = FatigueANN()

# Build models
cnn.build()
ann.build()

# Try loading pretrained weights if they exist
CNN_WEIGHTS = os.path.join('models', 'cnn_weights.h5')
ANN_WEIGHTS = os.path.join('models', 'ann_weights.h5')
cnn.load(CNN_WEIGHTS)
ann.load(ANN_WEIGHTS)

# MediaPipe setup
if MP_AVAILABLE:
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,          # We only care about the driver
        refine_landmarks=True,    # More precise eye + lip landmarks
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
else:
    face_mesh = None

# ──────────────────────────────────────────────────────────────────────────────
#  DETECTION THREAD
#  Runs in background, updates global state each frame
# ──────────────────────────────────────────────────────────────────────────────

def detection_loop():
    """
    Main per-frame processing loop.
    Runs in a separate thread so it doesn't block Flask's HTTP responses.

    Pipeline per frame:
      1. Read frame from webcam (OpenCV)
      2. Run MediaPipe FaceMesh
      3. If face found:
         a. Extract eye/mouth landmarks → EAR, MAR
         b. Update PERCLOS tracker and Blink tracker
         c. Estimate head pose (pitch, yaw, roll)
         d. Crop eye region → run CNN → get closed probability
         e. Assemble 6 features → run ANN → get fatigue score
         f. Run alert manager → get alert level
      4. Encode frame to JPEG → store as base64 for frontend
      5. Update global state dict
    """
    frame_count = 0
    fps_timer   = time.time()

    while True:
        with state_lock:
            if not state["running"]:
                break
            if state["paused"]:
                time.sleep(0.05)
                continue
            cap = state["cap"]
            ear_thresh = state["ear_threshold"]

        if cap is None or not cap.isOpened():
            time.sleep(0.05)
            continue

        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        frame_count += 1
        now = time.time()

        # ── FPS calculation ──
        if now - fps_timer >= 1.0:
            with state_lock:
                state["fps"] = frame_count
            frame_count = 0
            fps_timer = now

        h, w = frame.shape[:2]
        display = frame.copy()

        # ── Default values if no face detected ──
        ear = 0.0; mar = 0.0; perclos = 0.0
        blink_rate = 0; pitch = 0.0; yaw = 0.0; roll = 0.0
        cnn_prob = 0.0; fatigue = 0.0
        face_found = False

        # ── MediaPipe FaceMesh ────────────────────────────────────────────
        if MP_AVAILABLE and face_mesh is not None:
            # MediaPipe expects RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = face_mesh.process(rgb)
            rgb.flags.writeable = True

            if results.multi_face_landmarks:
                face_lm = results.multi_face_landmarks[0]
                face_found = True

                # ── Step 1: EAR ──────────────────────────────────────────
                left_eye_pts  = extract_eye_landmarks(face_lm, w, h, 'left')
                right_eye_pts = extract_eye_landmarks(face_lm, w, h, 'right')
                ear = compute_avg_ear(left_eye_pts, right_eye_pts)

                # Update EAR threshold from settings
                perclos_tracker.ear_threshold = ear_thresh
                blink_tracker.ear_threshold   = ear_thresh

                # ── Step 2: MAR ──────────────────────────────────────────
                mouth_pts = extract_mouth_landmarks(face_lm, w, h)
                mar = compute_mar(mouth_pts)

                # ── Step 3: PERCLOS ──────────────────────────────────────
                perclos_tracker.update(ear)
                perclos = perclos_tracker.get_perclos()

                # ── Step 4: Blink detection ──────────────────────────────
                blink_happened = blink_tracker.update(ear)
                if blink_happened:
                    with state_lock:
                        state["total_blinks"] += 1
                blink_rate = blink_tracker.get_blink_rate()
                consec     = blink_tracker.get_consecutive_closed_frames()

                # ── Step 5: Head pose ─────────────────────────────────────
                pose = estimate_head_pose(face_lm, w, h)
                pitch = pose["pitch"]
                yaw   = pose["yaw"]
                roll  = pose["roll"]

                # ── Step 6: CNN eye classification ────────────────────────
                left_crop  = crop_eye_region(frame, left_eye_pts, padding=12)
                right_crop = crop_eye_region(frame, right_eye_pts, padding=12)
                # Use the more closed eye (worst case) for CNN
                prob_l = cnn.predict(left_crop,  ear_fallback=ear)
                prob_r = cnn.predict(right_crop, ear_fallback=ear)
                cnn_prob = max(prob_l, prob_r)

                # ── Step 7: ANN fatigue score ─────────────────────────────
                # Blend CNN probability into EAR for the ANN
                # If CNN and EAR agree, confidence is high
                # If they disagree, take the average
                effective_ear = ear * (1.0 - 0.3 * cnn_prob)

                fatigue = ann.predict(
                    ear=effective_ear,
                    perclos=perclos,
                    blink_rate=blink_rate,
                    head_pitch=pitch,
                    head_yaw=yaw,
                    mar=mar,
                )

                # Sensitivity adjustment
                with state_lock:
                    sens = state["sensitivity"]
                mult = {1: 0.75, 2: 1.0, 3: 1.25}[sens]
                fatigue = min(fatigue * mult, 100.0)

                # ── Step 8: Alert level ────────────────────────────────────
                alert_result = alert_manager.update(
                    fatigue_score=fatigue,
                    ear=ear,
                    perclos=perclos,
                    consecutive_closed=consec,
                )

                if alert_result["is_new_alert"] and alert_result["level"] > 0:
                    with state_lock:
                        state["total_alerts"] += 1

                # ── Step 9: Draw overlay on frame ─────────────────────────
                with state_lock:
                    show_overlay = state["show_overlay"]

                if show_overlay:
                    display = draw_overlay(
                        display, face_lm, w, h,
                        ear, perclos, fatigue,
                        alert_result["level"],
                        pose, left_eye_pts, right_eye_pts
                    )

        # ── Encode frame as JPEG → base64 ────────────────────────────────
        _, jpeg = cv2.imencode('.jpg', display, [cv2.IMWRITE_JPEG_QUALITY, 70])
        frame_b64 = base64.b64encode(jpeg.tobytes()).decode('utf-8')

        # ── Update global state ────────────────────────────────────────────
        alert_info = alert_manager.current_level
        level_info = {
            0: ("#00C68A", "Normal",                  "All metrics normal — driver is alert"),
            1: ("#FFAA00", "Mild Fatigue",             "Early drowsiness signs detected — stay alert"),
            2: ("#FF7800", "Moderate Drowsiness",      "Significant drowsiness — consider a break"),
            3: ("#FF4D4D", "CRITICAL — Microsleep",    "SEVERE — Pull over immediately!"),
        }
        col, name, desc = level_info.get(alert_info, level_info[0])

        breakdown = ann.get_feature_breakdown(ear, perclos, blink_rate, pitch, yaw, mar)

        with state_lock:
            state.update({
                "ear":           round(ear, 4),
                "mar":           round(mar, 4),
                "perclos":       round(perclos, 2),
                "blink_rate":    int(blink_rate),
                "head_pitch":    round(pitch, 2),
                "head_yaw":      round(yaw, 2),
                "head_roll":     round(roll, 2),
                "cnn_prob":      round(cnn_prob, 4),
                "fatigue_score": round(fatigue, 1),
                "alert_level":   alert_info,
                "alert_name":    name,
                "alert_desc":    desc,
                "alert_color":   col,
                "face_detected": face_found,
                "frame_b64":     frame_b64,
                "breakdown":     breakdown,
            })

        time.sleep(0.03)   # ~30 fps max

    print("[Detection] Thread exited.")


def draw_overlay(frame, face_lm, w, h, ear, perclos, fatigue, level,
                 pose, left_pts, right_pts):
    """
    Draw face mesh, eye contours, metrics text, and head arrow on frame.
    """
    colors = [(0,198,138), (0,170,255), (0,120,255), (77,77,255)]  # BGR
    color  = colors[min(level, 3)]

    # ── Eye contours ──
    def draw_contour(pts, c):
        if pts and len(pts) >= 6:
            for i in range(len(pts)):
                p1 = (int(pts[i][0]), int(pts[i][1]))
                p2 = (int(pts[(i+1)%len(pts)][0]), int(pts[(i+1)%len(pts)][1]))
                cv2.line(frame, p1, p2, c, 1)

    eye_color = (77,77,255) if ear < 0.25 else (0,198,138)
    draw_contour(left_pts,  eye_color)
    draw_contour(right_pts, eye_color)

    # ── Head direction arrow (nose) ──
    rvec = pose.get("rvec")
    tvec = pose.get("tvec")
    if rvec is not None:
        from utils.head_pose import get_nose_direction_arrow
        tip, end = get_nose_direction_arrow(face_lm, w, h, rvec, tvec)
        if tip and end:
            cv2.arrowedLine(frame, tip, end, color, 2, tipLength=0.3)

    # ── Metrics overlay ──
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (240, 90), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

    cv2.putText(frame, f"EAR:    {ear:.3f}",      (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.putText(frame, f"PERCLOS:{perclos:.1f}%",  (14, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.putText(frame, f"Fatigue:{fatigue:.0f}/100",(14, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.putText(frame, f"Yaw:{pose['yaw']:.0f}  Pitch:{pose['pitch']:.0f}", (14, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # ── Alert label ──
    labels = ["NORMAL", "MILD FATIGUE", "MODERATE", "CRITICAL"]
    lbl = labels[min(level, 3)]
    cv2.putText(frame, lbl, (w - 150, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    return frame


# ──────────────────────────────────────────────────────────────────────────────
#  FLASK ROUTES
# ──────────────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Serve the main HTML page."""
    return render_template('index.html',
                           cnn_info=cnn.get_summary(),
                           ann_info=ann.get_summary())


@app.route('/api/start', methods=['POST'])
def api_start():
    """
    Start the webcam and detection thread.
    Called when user clicks 'Start Camera' button.
    """
    with state_lock:
        if state["running"]:
            return jsonify({"ok": False, "msg": "Already running"})

    cap = cv2.VideoCapture(0)   # 0 = default webcam
    if not cap.isOpened():
        return jsonify({"ok": False, "msg": "Could not open camera. Check it is connected and not in use."})

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Reset all trackers for new session
    perclos_tracker.reset()
    blink_tracker.reset()
    alert_manager.reset()

    with state_lock:
        state["cap"]           = cap
        state["running"]       = True
        state["paused"]        = False
        state["session_start"] = time.time()
        state["total_alerts"]  = 0
        state["total_blinks"]  = 0
        state["frame_b64"]     = None

    t = threading.Thread(target=detection_loop, daemon=True)
    t.start()
    with state_lock:
        state["thread"] = t

    return jsonify({"ok": True, "msg": "Camera started"})


@app.route('/api/stop', methods=['POST'])
def api_stop():
    """
    Stop the webcam and detection thread.
    Called when user clicks 'Stop Camera' button.
    """
    with state_lock:
        state["running"] = False
        cap = state["cap"]
        state["cap"] = None

    if cap:
        cap.release()

    with state_lock:
        state.update({
            "frame_b64":     None,
            "face_detected": False,
            "alert_level":   0,
            "alert_name":    "Session Ended",
            "alert_desc":    "Camera stopped",
            "alert_color":   "#00C68A",
        })

    return jsonify({"ok": True, "msg": "Camera stopped"})


@app.route('/api/pause', methods=['POST'])
def api_pause():
    """Toggle pause/resume."""
    with state_lock:
        state["paused"] = not state["paused"]
        paused = state["paused"]
    return jsonify({"ok": True, "paused": paused})


@app.route('/api/metrics')
def api_metrics():
    """
    Return current metrics as JSON.
    Called by the frontend every 100ms via polling.

    Response includes all computed metrics plus the feature breakdown
    showing how much each signal contributes to the fatigue score.
    """
    with state_lock:
        session_elapsed = 0
        if state["session_start"]:
            session_elapsed = int(time.time() - state["session_start"])

        return jsonify({
            "running":       state["running"],
            "paused":        state["paused"],
            "face_detected": state["face_detected"],
            "fps":           state["fps"],

            # Core metrics
            "ear":           state["ear"],
            "mar":           state["mar"],
            "perclos":       state["perclos"],
            "blink_rate":    state["blink_rate"],
            "head_pitch":    state["head_pitch"],
            "head_yaw":      state["head_yaw"],
            "head_roll":     state["head_roll"],
            "cnn_prob":      state["cnn_prob"],
            "fatigue_score": state["fatigue_score"],
            "alert_level":   state["alert_level"],
            "alert_name":    state["alert_name"],
            "alert_desc":    state["alert_desc"],
            "alert_color":   state["alert_color"],

            # Session stats
            "session_seconds": session_elapsed,
            "total_alerts":  state["total_alerts"],
            "total_blinks":  state["total_blinks"],

            # ANN feature breakdown (for explanation panel)
            "breakdown":     state["breakdown"],
        })


@app.route('/api/frame')
def api_frame():
    """
    Return current processed frame as base64 JPEG.
    Called by the frontend every 100ms to update the video display.
    """
    with state_lock:
        b64 = state["frame_b64"]

    if b64 is None:
        return jsonify({"frame": None})
    return jsonify({"frame": b64})


@app.route('/api/log')
def api_log():
    """Return alert event log."""
    return jsonify({"events": alert_manager.get_event_log()})


@app.route('/api/settings', methods=['POST'])
def api_settings():
    """
    Update detection settings.
    Called when user changes sliders / toggles in the settings panel.
    """
    data = request.json
    with state_lock:
        if "ear_threshold" in data:
            state["ear_threshold"] = float(data["ear_threshold"])
        if "show_overlay" in data:
            state["show_overlay"] = bool(data["show_overlay"])
        if "sensitivity" in data:
            state["sensitivity"] = int(data["sensitivity"])

    return jsonify({"ok": True})


@app.route('/api/snapshot', methods=['POST'])
def api_snapshot():
    """
    Save current frame as a PNG snapshot file.
    Called when user clicks Snapshot button.
    """
    with state_lock:
        cap = state["cap"]
        fatigue = state["fatigue_score"]
        level   = state["alert_level"]

    if cap is None:
        return jsonify({"ok": False, "msg": "Camera not running"})

    ret, frame = cap.read()
    if not ret:
        return jsonify({"ok": False, "msg": "Could not capture frame"})

    # Add info overlay to snapshot
    ts = time.strftime("%Y%m%d_%H%M%S")
    label = f"DrowsyGuard | Fatigue: {fatigue:.0f}/100 | {time.strftime('%H:%M:%S')}"
    cv2.rectangle(frame, (0, frame.shape[0]-36), (frame.shape[1], frame.shape[0]), (0,0,0), -1)
    cv2.putText(frame, label, (8, frame.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 198, 138), 1)

    path = os.path.join('static', 'snapshots', f'snap_{ts}.jpg')
    cv2.imwrite(path, frame)

    return jsonify({"ok": True, "path": f"/static/snapshots/snap_{ts}.jpg"})


@app.route('/api/reset', methods=['POST'])
def api_reset():
    """Reset session counters and trackers."""
    perclos_tracker.reset()
    blink_tracker.reset()
    alert_manager.reset()

    with state_lock:
        state["session_start"] = time.time()
        state["total_alerts"]  = 0
        state["total_blinks"]  = 0

    return jsonify({"ok": True})


@app.route('/api/status')
def api_status():
    """Return system info (model status, versions)."""
    return jsonify({
        "mediapipe": MP_AVAILABLE,
        "cnn":       cnn.get_summary(),
        "ann":       ann.get_summary(),
        "flask":     True,
    })


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "═"*50)
    print("  DrowsyGuard — Starting server")
    print("  Open: http://localhost:5000")
    print("═"*50 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
