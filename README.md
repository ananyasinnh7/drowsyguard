# DrowsyGuard — Python + AI Backend + Full Website

## Project Structure
```
drowsyguard/
├── app.py                  ← Flask web server (entry point)
├── requirements.txt        ← Python dependencies
├── utils/
│   ├── ear.py              ← EAR / MAR geometry calculations
│   ├── perclos.py          ← PERCLOS rolling window tracker
│   ├── head_pose.py        ← Head pose estimation (3D)
│   └── alert.py            ← Alert level logic
├── models/
│   ├── cnn_model.py        ← CNN definition (MobileNetV2-based)
│   ├── ann_model.py        ← ANN fusion model definition
│   └── train.py            ← Training script (optional)
├── templates/
│   └── index.html          ← Full frontend (served by Flask)
└── static/
    └── (auto-created for snapshots)
```

## Setup Instructions

### 1. Install Python 3.9+
https://www.python.org/downloads/

### 2. Create virtual environment (recommended)
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the server
```bash
python app.py
```

### 5. Open browser
```
http://localhost:5000
```

## How Each AI Component Works

### MediaPipe (Face Landmark Detection)
- Detects 468 3D face landmarks per frame in real time
- Used to locate eye corners, mouth corners, nose tip
- Runs at 30+ fps on CPU

### EAR — Eye Aspect Ratio (utils/ear.py)
- Geometric formula using 6 eye landmark points
- EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
- When EAR < threshold (0.25), eye is considered closed

### PERCLOS (utils/perclos.py)
- Rolling 60-second window tracking % time eyes closed
- PERCLOS > 35% = drowsy threshold

### CNN (models/cnn_model.py)
- MobileNetV2 architecture
- Input: 64x64 cropped eye region image
- Output: probability of eye being closed [0.0 - 1.0]
- Trained on NTHU-DDD / custom eye dataset

### ANN (models/ann_model.py)
- 3-layer fully connected network
- Input: [EAR, PERCLOS, blink_rate, head_pitch, head_yaw, MAR]
- Output: fatigue score 0-100
- Combines all signals into one unified drowsiness score

### Head Pose (utils/head_pose.py)
- Uses solvePnP with 3D face model
- Estimates pitch (nod), yaw (turn), roll (tilt) angles

### Alert Logic (utils/alert.py)
- Level 0 (Normal):   fatigue < 25
- Level 1 (Mild):     fatigue 25-44
- Level 2 (Moderate): fatigue 45-69
- Level 3 (Critical): fatigue >= 70
