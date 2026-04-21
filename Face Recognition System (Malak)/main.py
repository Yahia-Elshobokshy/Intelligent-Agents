"""
main.py  —  Face Recognition + Anti-Spoof server
Supports TWO input modes (toggle with MODE constant):
  "webcam"  — uses local camera (development / testing)
  "esp32"   — receives JPEG frames from ESP32-CAM over HTTP

Run:  python main.py
ESP32 posts frames to:  http://<your-pc-ip>:5000/frame
"""

import cv2
import face_recognition
import numpy as np
import pickle
import threading
import time
from flask import Flask, request, jsonify, Response
from anti_spoof import check_liveness

# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════
MODE              = "webcam"   # "webcam" | "esp32"
WEBCAM_INDEX      = 0
RECOGNITION_SCALE = 0.5        # downscale for speed
MATCH_TOLERANCE   = 0.50       # lower = stricter face matching
SPOOF_INTERVAL    = 8          # re-check liveness every N frames
FLASK_PORT        = 5000

# ═══════════════════════════════════════════════════════════════════════
# Load encodings
# ═══════════════════════════════════════════════════════════════════════
print("[BOOT] Loading encodings…")
with open("encodings.pkl", "rb") as f:
    known_encodings, known_names = pickle.load(f)
print(f"[BOOT] {len(known_encodings)} faces loaded for: {sorted(set(known_names))}")

# ═══════════════════════════════════════════════════════════════════════
# Shared state (used by ESP32 thread + display thread)
# ═══════════════════════════════════════════════════════════════════════
latest_frame_lock = threading.Lock()
latest_frame      = None          # raw BGR frame from ESP32

app = Flask(__name__)


# ───────────────────────────────────────────────────────────────────────
# ESP32 endpoint: POST /frame   body = raw JPEG bytes
# ───────────────────────────────────────────────────────────────────────
@app.route("/frame", methods=["POST"])
def receive_frame():
    global latest_frame
    jpeg_bytes = request.data
    nparr      = np.frombuffer(jpeg_bytes, np.uint8)
    frame      = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is not None:
        with latest_frame_lock:
            latest_frame = frame
    return jsonify({"status": "ok"}), 200


# ───────────────────────────────────────────────────────────────────────
# ESP32 endpoint: GET /decision  → returns last access decision as JSON
# ───────────────────────────────────────────────────────────────────────
decision_lock = threading.Lock()
last_decision = {"name": "Unknown", "is_live": False, "confidence": 0.0, "grant": False}

@app.route("/decision", methods=["GET"])
def get_decision():
    with decision_lock:
        return jsonify(last_decision)


# ═══════════════════════════════════════════════════════════════════════
# Core processing loop
# ═══════════════════════════════════════════════════════════════════════
def process_loop():
    global last_decision

    cap         = cv2.VideoCapture(WEBCAM_INDEX) if MODE == "webcam" else None
    frame_count = 0
    liveness_cache = {}     # face_id → (is_live, conf)

    def get_face_id(t, r, b, l):
        """Stable grid-snapped ID so we don't recompute every frame."""
        cy = ((t + b) // 2) // 40
        cx = ((l + r) // 2) // 40
        return (cy, cx)

    print(f"[RUN] Mode: {MODE}  — press ESC to quit")

    while True:
        # ── Grab frame ──────────────────────────────────────────────────
        if MODE == "webcam":
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
        else:
            with latest_frame_lock:
                frame = latest_frame
            if frame is None:
                time.sleep(0.02)
                continue
            frame = frame.copy()

        frame_count += 1

        # ── Downscale for detection ──────────────────────────────────────
        small = cv2.resize(frame, (0, 0), fx=RECOGNITION_SCALE, fy=RECOGNITION_SCALE)
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        # ── Face detection + encoding ────────────────────────────────────
        locations = face_recognition.face_locations(rgb, model="hog")
        encodings = face_recognition.face_encodings(rgb, locations)

        # Clear stale liveness cache every 60 frames
        if frame_count % 60 == 0:
            liveness_cache.clear()

        inv = int(1 / RECOGNITION_SCALE)
        best_result = {"name": "Unknown", "is_live": False,
                       "confidence": 0.0,  "grant": False}

        for (t, r, b, l), enc in zip(locations, encodings):
            # Scale coords back up
            t, r, b, l = t * inv, r * inv, b * inv, l * inv

            # ── Identity ─────────────────────────────────────────────────
            name = "Unknown"
            if known_encodings:
                dists    = face_recognition.face_distance(known_encodings, enc)
                best_idx = int(np.argmin(dists))
                if dists[best_idx] < MATCH_TOLERANCE:
                    name = known_names[best_idx]

            # ── Anti-spoof (cached, checked every SPOOF_INTERVAL frames) ─
            fid = get_face_id(t, r, b, l)
            if frame_count % SPOOF_INTERVAL == 0 or fid not in liveness_cache:
                is_live, conf, _ = check_liveness(frame, (l, t, r, b))
                liveness_cache[fid] = (is_live, conf)
            else:
                is_live, conf = liveness_cache[fid]

            # ── Decision ─────────────────────────────────────────────────
            grant = (name != "Unknown") and is_live

            # ── Draw ─────────────────────────────────────────────────────
            if grant:
                color = (0, 210, 0)
                status = "LIVE"
            elif name != "Unknown":
                color  = (0, 80, 220)
                status = "SPOOF"
            elif is_live:
                color  = (0, 140, 255)
                status = "LIVE"
            else:
                color  = (0, 0, 200)
                status = "SPOOF"

            conf_pct = int(conf * 100)
            label    = f"{name}  {status} {conf_pct}%"

            # Box
            cv2.rectangle(frame, (l, t), (r, b), color, 2)

            # Label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(frame, (l, t - th - 14), (l + tw + 8, t), color, -1)
            cv2.putText(frame, label, (l + 4, t - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            # Confidence bar
            bar_w  = r - l
            filled = int(bar_w * np.clip(conf, 0, 1))
            bar_c  = (0, 210, 0) if is_live else (0, 0, 200)
            cv2.rectangle(frame, (l, b + 4), (r, b + 12), (50, 50, 50), -1)
            cv2.rectangle(frame, (l, b + 4), (l + filled, b + 12), bar_c, -1)

            # Keep highest-confidence result as the access decision
            if conf > best_result["confidence"] or grant:
                best_result = {"name": name, "is_live": is_live,
                               "confidence": round(conf, 3), "grant": grant}

        # Push decision (ESP32 can poll /decision)
        with decision_lock:
            last_decision = best_result

        # ── Display ──────────────────────────────────────────────────────
        mode_label = "WEBCAM" if MODE == "webcam" else "ESP32-CAM"
        cv2.putText(frame, f"[{mode_label}]  ESC = quit",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Security System", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    if cap:
        cap.release()
    cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    if MODE == "esp32":
        # Flask runs in background thread; display loop on main thread
        flask_thread = threading.Thread(
            target=lambda: app.run(host="0.0.0.0", port=FLASK_PORT, threaded=True),
            daemon=True
        )
        flask_thread.start()
        print(f"[BOOT] Flask listening on port {FLASK_PORT}")

    process_loop()