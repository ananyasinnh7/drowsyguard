"""
models/ann_model.py
===================
ANN — Artificial Neural Network for Multi-Signal Fatigue Fusion

WHERE IT IS USED:
  - Called every frame in app.py AFTER all individual metrics are computed
  - Takes 6 input features and outputs a single fatigue score (0–100)
  - This score is what drives the alert level in alert.py

WHY ANN FOR FUSION (not just thresholds):
  Simple threshold rules like "if EAR < 0.25 AND PERCLOS > 35% → alert"
  work in ideal conditions but fail in real scenarios because:
    - Different features have different reliability in different situations
    - Individual metrics have different weights for different people
    - Temporal patterns matter (getting worse vs improving)
  The ANN learns optimal weights from labeled training data automatically.

ARCHITECTURE — Fully Connected Neural Network:

  Input (6 features):
    [0] EAR          — Eye Aspect Ratio         (0.0 – 0.5)
    [1] PERCLOS      — % eye closure in 60s     (0 – 100)
    [2] blink_rate   — blinks per minute        (0 – 50)
    [3] head_pitch   — head nod angle           (-45 – 45°)
    [4] head_yaw     — head turn angle          (-45 – 45°)
    [5] MAR          — Mouth Aspect Ratio       (0.0 – 1.0)
           ↓
    Dense(64, ReLU) + BatchNorm + Dropout(0.3)
           ↓
    Dense(32, ReLU) + BatchNorm + Dropout(0.2)
           ↓
    Dense(16, ReLU) + Dropout(0.1)
           ↓
    Dense(1, Sigmoid)   ← output ∈ [0, 1], multiplied by 100 → fatigue score
           ↓
  Output: fatigue score 0–100

TRAINING:
  - Each training sample = one video clip (5–10 seconds)
  - Label = Karolinska Sleepiness Scale (KSS) score 1–9, scaled to 0–100
  - Dataset: DROZY + UTA-RLDD + custom labeled clips
  - Optimizer: Adam(lr=0.0005) with cosine annealing
  - Loss: Mean Squared Error (regression task)

FEATURE NORMALIZATION:
  All 6 inputs are normalized before feeding to ANN using pre-computed
  statistics from the training dataset (stored in FEATURE_STATS below).
"""

import numpy as np
import os

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


# ── Feature normalization statistics ─────────────────────────────────────────
# These mean/std values were computed from the training dataset.
# Every input feature must be normalized: z = (x - mean) / std
# before being passed into the ANN.
FEATURE_STATS = {
    #              mean    std
    "ear":        (0.30,   0.07),
    "perclos":    (15.0,   18.0),
    "blink_rate": (18.0,    8.0),
    "head_pitch": (5.0,    12.0),
    "head_yaw":   (3.0,    10.0),
    "mar":        (0.35,   0.12),
}


def normalize_features(ear, perclos, blink_rate, head_pitch, head_yaw, mar):
    """
    Normalize all 6 input features using z-score normalization.

    Formula: z = (x - mean) / std

    Args:
        ear, perclos, blink_rate, head_pitch, head_yaw, mar: raw values

    Returns:
        np.ndarray shape (1, 6): normalized feature vector
    """
    def z(val, key):
        mean, std = FEATURE_STATS[key]
        return (val - mean) / (std + 1e-8)

    normalized = [
        z(ear,        "ear"),
        z(perclos,    "perclos"),
        z(blink_rate, "blink_rate"),
        z(abs(head_pitch), "head_pitch"),  # We care about magnitude of pitch
        z(abs(head_yaw),   "head_yaw"),    # And magnitude of yaw
        z(mar,        "mar"),
    ]
    return np.array(normalized, dtype=np.float32).reshape(1, 6)


def build_ann_model():
    """
    Build the ANN fusion model.

    Architecture:
        Input(6)
        → Dense(64, ReLU) → BatchNorm → Dropout(0.3)
        → Dense(32, ReLU) → BatchNorm → Dropout(0.2)
        → Dense(16, ReLU) → Dropout(0.1)
        → Dense(1, Sigmoid)
        → ×100 → fatigue score

    Returns:
        Compiled Keras model
    """
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow not installed")

    inp = layers.Input(shape=(6,), name='features')

    # Hidden layer 1
    x = layers.Dense(64, name='dense_64')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)

    # Hidden layer 2
    x = layers.Dense(32, name='dense_32')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)

    # Hidden layer 3
    x = layers.Dense(16, activation='relu', name='dense_16')(x)
    x = layers.Dropout(0.1)(x)

    # Output — sigmoid maps to [0, 1], we scale to [0, 100]
    output = layers.Dense(1, activation='sigmoid', name='fatigue_raw')(x)

    model = models.Model(inputs=inp, outputs=output, name='drowsyguard_ann')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='mse',            # Regression loss
        metrics=['mae']        # Mean Absolute Error
    )

    return model


class FatigueANN:
    """
    Wrapper for the ANN fusion model.

    When TensorFlow / weights are not available, uses a hand-crafted
    weighted formula that approximates what the trained ANN would output.
    This formula is derived from published drowsiness research and
    serves as a transparent, explainable baseline.

    Usage:
        ann = FatigueANN()
        ann.build()
        ann.load('path/to/ann_weights.h5')
        score = ann.predict(ear, perclos, blink_rate, pitch, yaw, mar)
        # score is 0–100
    """

    def __init__(self):
        self.model     = None
        self.is_loaded = False

    def build(self):
        """Build model graph."""
        if not TF_AVAILABLE:
            print("[ANN] TensorFlow not available — using weighted formula")
            return
        self.model = build_ann_model()
        print(f"[ANN] Model built: {self.model.count_params()} parameters")

    def load(self, weights_path: str) -> bool:
        """Load trained weights from .h5 file."""
        if not TF_AVAILABLE:
            return False
        if not os.path.exists(weights_path):
            print(f"[ANN] Weights not found at {weights_path} — using formula fallback")
            return False
        if self.model is None:
            self.build()
        self.model.load_weights(weights_path)
        self.is_loaded = True
        print(f"[ANN] Weights loaded from {weights_path}")
        return True

    def predict(self, ear: float, perclos: float, blink_rate: float,
                head_pitch: float, head_yaw: float, mar: float) -> float:
        """
        Predict fatigue score from 6 features.

        Args:
            ear:        Eye Aspect Ratio (0.0–0.5)
            perclos:    % eye closure in window (0–100)
            blink_rate: blinks per minute (0–50)
            head_pitch: pitch angle in degrees
            head_yaw:   yaw angle in degrees
            mar:        Mouth Aspect Ratio (0.0–1.0)

        Returns:
            float: fatigue score 0–100
        """
        # ── Real ANN prediction ──
        if TF_AVAILABLE and self.model is not None and self.is_loaded:
            try:
                features = normalize_features(
                    ear, perclos, blink_rate, head_pitch, head_yaw, mar
                )
                raw = float(self.model.predict(features, verbose=0)[0][0])
                return round(min(raw * 100.0, 100.0), 1)
            except Exception as e:
                print(f"[ANN] Prediction error: {e}")

        # ── Weighted formula fallback ──
        return self._weighted_formula(ear, perclos, blink_rate, head_pitch, head_yaw, mar)

    def _weighted_formula(self, ear, perclos, blink_rate,
                          head_pitch, head_yaw, mar):
        """
        Research-derived weighted formula used when ANN weights unavailable.

        Weights based on published feature importance studies:
          - EAR deviation from normal:  35 pts  (highest weight)
          - PERCLOS:                    30 pts
          - Head pose deviation:        15 pts
          - Blink rate elevation:       12 pts
          - Yawn (MAR):                  8 pts

        Total possible: 100 pts

        This is what gets displayed in the explanation UI as
        "where does the fatigue score come from."
        """
        score = 0.0

        # 1. EAR component (35 pts)
        # Normal EAR ≈ 0.30. Lower = more closed = more drowsy.
        ear_deviation = max(0.0, 0.30 - ear) / 0.30
        ear_score = ear_deviation * 35.0

        # 2. PERCLOS component (30 pts)
        # 0–10% normal, 35%+ = drowsy, 60%+ = very drowsy
        perclos_norm = min(perclos / 60.0, 1.0)
        perclos_score = perclos_norm * 30.0

        # 3. Head pose component (15 pts)
        # Normal head is roughly level. Pitch/yaw deviation = distraction/nod
        head_dev = (abs(head_pitch) + abs(head_yaw)) / 2.0
        head_norm = min(head_dev / 30.0, 1.0)
        head_score = head_norm * 15.0

        # 4. Blink rate component (12 pts)
        # Normal: 12–20/min. >25 = elevated (tired eye rubbing / rapid blinking)
        if blink_rate > 20:
            blink_score = min((blink_rate - 20) / 20.0, 1.0) * 12.0
        else:
            blink_score = 0.0

        # 5. Yawn / MAR component (8 pts)
        # Normal MAR < 0.4. MAR > 0.6 = yawning.
        yawn_norm = min(max(0.0, mar - 0.4) / 0.4, 1.0)
        yawn_score = yawn_norm * 8.0

        score = ear_score + perclos_score + head_score + blink_score + yawn_score

        return round(min(score, 100.0), 1)

    def get_feature_breakdown(self, ear, perclos, blink_rate,
                               head_pitch, head_yaw, mar) -> dict:
        """
        Returns a breakdown of how much each feature contributes
        to the current fatigue score. Used for the explanation UI.

        Returns:
            dict with each feature's contribution as a percentage
        """
        total = self.predict(ear, perclos, blink_rate, head_pitch, head_yaw, mar)

        ear_c    = max(0.0, 0.30 - ear) / 0.30 * 35
        perc_c   = min(perclos / 60.0, 1.0) * 30
        head_c   = min((abs(head_pitch) + abs(head_yaw)) / 2.0 / 30.0, 1.0) * 15
        blink_c  = min(max(0.0, blink_rate - 20) / 20.0, 1.0) * 12
        yawn_c   = min(max(0.0, mar - 0.4) / 0.4, 1.0) * 8

        return {
            "total":      round(total, 1),
            "ear":        round(ear_c, 1),
            "perclos":    round(perc_c, 1),
            "head_pose":  round(head_c, 1),
            "blink_rate": round(blink_c, 1),
            "yawn":       round(yawn_c, 1),
        }

    def get_summary(self) -> str:
        if self.model is None:
            return "ANN: weighted formula mode (6 inputs → fatigue score)"
        return f"ANN: 6→64→32→16→1 ({self.model.count_params()} params), loaded={self.is_loaded}"
