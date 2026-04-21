"""
anti_spoof.py  —  Lightweight texture-based liveness detection
No external model download required. Works on CPU in real time.

How it works:
  Real faces  → high-frequency skin micro-texture (pores, hair, depth variation)
  Printed/screen photos → flat, low-variance texture after Laplacian filter
  We measure Local Binary Pattern (LBP) variance + Laplacian variance
  across the face crop. Below thresholds = spoof.
"""

import cv2
import numpy as np


# ── Tuning constants ────────────────────────────────────────────────────────
# Raise LAP_THRESH if real faces are flagged as spoof (camera is blurry)
# Lower  LAP_THRESH if printed photos pass as real
LAP_THRESH  = 80.0   # Laplacian variance  — measures sharpness / texture depth
LBP_THRESH  = 0.45   # LBP entropy         — measures micro-texture complexity
EDGE_THRESH = 12.0   # Edge density        — printed photos often have moiré patterns


def _laplacian_variance(gray_crop: np.ndarray) -> float:
    """Higher = more texture depth (real skin > flat photo)."""
    return float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())


def _lbp_entropy(gray_crop: np.ndarray) -> float:
    """
    Simplified Local Binary Pattern entropy.
    Real skin has rich, varied LBP patterns → higher entropy.
    Flat printed image → lower entropy.
    """
    h, w   = gray_crop.shape
    radius = 1
    n_pts  = 8

    lbp = np.zeros_like(gray_crop, dtype=np.uint8)
    for i in range(radius, h - radius):
        for j in range(radius, w - radius):
            center = int(gray_crop[i, j])
            code   = 0
            # Sample 8 neighbours in a circle
            neighbours = [
                gray_crop[i - 1, j - 1], gray_crop[i - 1, j], gray_crop[i - 1, j + 1],
                gray_crop[i,     j + 1],
                gray_crop[i + 1, j + 1], gray_crop[i + 1, j], gray_crop[i + 1, j - 1],
                gray_crop[i,     j - 1],
            ]
            for k, nb in enumerate(neighbours):
                if int(nb) >= center:
                    code |= (1 << k)
            lbp[i, j] = code

    # Compute normalised histogram entropy
    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
    hist    = hist / (hist.sum() + 1e-7)
    entropy = -np.sum(hist * np.log2(hist + 1e-7))
    return float(entropy / 8.0)   # normalise to 0-1


def _edge_density(gray_crop: np.ndarray) -> float:
    """Canny edge density — printed photos often have sharp halftone edges."""
    edges = cv2.Canny(gray_crop, 50, 150)
    return float(edges.mean())


def check_liveness(frame: np.ndarray, face_box: tuple) -> tuple:
    """
    Parameters
    ----------
    frame    : full BGR frame from OpenCV
    face_box : (left, top, right, bottom) in frame pixel coords

    Returns
    -------
    (is_live: bool, confidence: float 0.0–1.0, label: str)
    """
    left, top, right, bottom = face_box

    # Add 10 % margin so we capture forehead / chin texture too
    h_frame, w_frame = frame.shape[:2]
    margin_x = max(10, int((right - left)  * 0.10))
    margin_y = max(10, int((bottom - top)  * 0.10))
    x1 = max(0, left   - margin_x)
    y1 = max(0, top    - margin_y)
    x2 = min(w_frame, right  + margin_x)
    y2 = min(h_frame, bottom + margin_y)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return False, 0.0, "Invalid"

    # Normalise crop size for consistent scoring
    crop = cv2.resize(crop, (128, 128))
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # ── Three independent cues ──────────────────────────────────────────────
    lap_var    = _laplacian_variance(gray)
    lbp_ent    = _lbp_entropy(gray)
    edge_dens  = _edge_density(gray)

    lap_pass  = lap_var   > LAP_THRESH
    lbp_pass  = lbp_ent   > LBP_THRESH
    edge_pass = edge_dens > EDGE_THRESH

    votes = int(lap_pass) + int(lbp_pass) + int(edge_pass)

    # Confidence: weighted blend normalised to 0–1
    conf = np.clip(
        0.40 * (lap_var   / (LAP_THRESH  * 2.0)) +
        0.35 * (lbp_ent   / (LBP_THRESH  * 2.0)) +
        0.25 * (edge_dens / (EDGE_THRESH * 2.0)),
        0.0, 1.0
    )

    # Need at least 2 of 3 cues to pass
    is_live = votes >= 2
    label   = "LIVE" if is_live else "SPOOF"

    return is_live, float(conf), label


# ── Quick standalone test ───────────────────────────────────────────────────
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    import face_recognition

    print("Anti-spoof test — press ESC to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        locs  = face_recognition.face_locations(rgb, model="hog")

        for (t, r, b, l) in locs:
            t, r, b, l = t * 2, r * 2, b * 2, l * 2
            is_live, conf, label = check_liveness(frame, (l, t, r, b))
            color = (0, 200, 0) if is_live else (0, 0, 220)
            cv2.rectangle(frame, (l, t), (r, b), color, 2)
            cv2.putText(frame, f"{label} {int(conf*100)}%",
                        (l, t - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Anti-Spoof Test", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()