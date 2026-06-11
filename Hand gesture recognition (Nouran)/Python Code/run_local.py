import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import time
from collections import deque


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG — adjust paths to match your project
# ═══════════════════════════════════════════════════════════════════════════════

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "AI Model")
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pth")
LANDMARKER_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")

# If the landmarker file doesn't exist locally, download it
if not os.path.exists(LANDMARKER_PATH):
    import urllib.request
    print("Downloading hand_landmarker.task...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    os.makedirs(BASE_DIR, exist_ok=True)
    urllib.request.urlretrieve(url, LANDMARKER_PATH)
    print("Downloaded!")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ═══════════════════════════════════════════════════════════════════════════════
# CLASS NAMES (18 HaGRID classes — must match training order)
# ═══════════════════════════════════════════════════════════════════════════════

class_names = [
    'call', 'dislike', 'fist', 'four', 'like', 'mute', 'ok', 'one', 'palm',
    'peace', 'peace_inverted', 'rock', 'stop', 'stop_inverted', 'three',
    'three2', 'two_up', 'two_up_inverted'
]
class_to_idx = {name: idx for idx, name in enumerate(class_names)}
idx_to_class = {idx: name for name, idx in class_to_idx.items()}

SECURITY_ACTIONS = {
    'call': '📞 CALLING FOR HELP',
    'dislike': '❌ ACCESS DENIED',
    'fist': '🚨 ALERT TRIGGERED',
    'four': '4️⃣ SIGNAL FOUR',
    'like': '✅ ACCESS GRANTED',
    'mute': '🔇 SYSTEM MUTED',
    'ok': '✅ ALL CLEAR',
    'one': '☝️ SIGNAL ONE',
    'palm': '✋ HALT — STOP',
    'peace': '✌️ PEACE — STANDBY',
    'peace_inverted': '⚠️ WARNING',
    'rock': '🤘 SIGNAL ROCK',
    'stop': '🛑 SYSTEM STOP',
    'stop_inverted': '🔄 SYSTEM RESUME',
    'three': '3️⃣ SIGNAL THREE',
    'three2': '3️⃣ SIGNAL THREE ALT',
    'two_up': '✌️ SIGNAL TWO',
    'two_up_inverted': '⚠️ CAUTION',
}

# ═══════════════════════════════════════════════════════════════════════════════
# TRANSFORMS (must match training)
# ═══════════════════════════════════════════════════════════════════════════════

val_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
])

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD MODEL
# ═══════════════════════════════════════════════════════════════════════════════

print("Loading model...")
model = efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()
print(f"✅ Model loaded from {MODEL_PATH}")

# ═══════════════════════════════════════════════════════════════════════════════
# HAND DETECTOR (same logic as your notebook)
# ═══════════════════════════════════════════════════════════════════════════════

print("Initializing hand detector...")
base_options = python.BaseOptions(model_asset_path=LANDMARKER_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.2,
    min_hand_presence_confidence=0.2,
    min_tracking_confidence=0.2
)
hand_detector = vision.HandLandmarker.create_from_options(options)
print("✅ Hand detector ready")

def detect_and_crop_hand(img_pil):
    """
    Detect hand and crop. Same logic as your notebook:
      - Tries original orientation
      - Falls back to flipped if no hand found
      - Returns (cropped_pil_image, hand_found_bool)
    """
    w, h = img_pil.size

    def _try_detect(image_pil):
        img_rgb = np.array(image_pil, dtype=np.uint8)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        return hand_detector.detect(mp_image)

    result = _try_detect(img_pil)
    flipped = False

    if not result.hand_landmarks:
        img_flipped = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
        result = _try_detect(img_flipped)
        if result.hand_landmarks:
            flipped = True
            img_pil = img_flipped

    if result.hand_landmarks:
        landmarks = result.hand_landmarks[0]
        x_coords = [lm.x for lm in landmarks]
        y_coords = [lm.y for lm in landmarks]

        x_min = max(0.0, min(x_coords))
        y_min = max(0.0, min(y_coords))
        x_max = min(1.0, max(x_coords))
        y_max = min(1.0, max(y_coords))

        pad_x = (x_max - x_min) * 0.30
        pad_y = (y_max - y_min) * 0.30

        x1 = max(0, int((x_min - pad_x) * w))
        y1 = max(0, int((y_min - pad_y) * h))
        x2 = min(w, int((x_max + pad_x) * w))
        y2 = min(h, int((y_max + pad_y) * h))

        if x2 > x1 and y2 > y1:
            return img_pil.crop((x1, y1, x2, y2)), True

    return img_pil, False


# ═══════════════════════════════════════════════════════════════════════════════
# LOCAL OPENCV CAMERA LOOP (replaces Colab JavaScript + output.eval_js)
# ═══════════════════════════════════════════════════════════════════════════════

def run_inference_loop(max_frames=5000):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # CAP_DSHOW for Windows stability

    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("❌ Cannot open webcam. Check permissions.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    time.sleep(1.0)

    print("\n" + "═" * 60)
    print("  🔐 GESTURE SECURITY SYSTEM — LOCAL MODE")
    print("═" * 60)
    print("Controls: [Q] or [ESC] = Quit  |  [S] = Screenshot")
    print("═" * 60 + "\n")

    frame_count = 0
    fps_history = deque(maxlen=30)
    last_time = time.time()

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            continue

        # FPS calc
        now = time.time()
        fps_history.append(1.0 / (now - last_time))
        last_time = now
        avg_fps = np.mean(fps_history)

        # Convert BGR → RGB → PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        # ── Hand detection ──
        cropped, hand_found = detect_and_crop_hand(pil_img)

        display = frame.copy()
        h, w = display.shape[:2]

        if not hand_found:
            # ── NO HAND ──
            cv2.rectangle(display, (0, 0), (w, h), (68, 68, 68), 3)

            overlay = display.copy()
            cv2.rectangle(overlay, (20, 20), (w - 20, 120), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.8, display, 0.2, 0, display)

            cv2.putText(display, "SECURITY SYSTEM ACTIVE", (35, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(display, "NO GESTURE DETECTED", (35, 95),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.95, (136, 136, 136), 2, cv2.LINE_AA)
            cv2.putText(display, "Show your hand to the camera", (35, 125),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (102, 102, 102), 1, cv2.LINE_AA)

            cv2.circle(display, (w - 40, 40), 8, (0, 0, 255), -1)

        else:
            # ── HAND FOUND → RUN MODEL ──
            tensor = val_transform(cropped).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1)[0]

            top5_conf, top5_idx = probs.topk(5)
            results = [(idx_to_class[i.item()], c.item() * 100)
                       for i, c in zip(top5_idx, top5_conf)]

            top_gesture, top_conf = results[0]
            action = SECURITY_ACTIONS.get(top_gesture, '❓ UNKNOWN')

            # Color by confidence
            if top_conf > 80:
                color = (0, 255, 0)
            elif top_conf > 55:
                color = (0, 170, 255)
            else:
                color = (0, 80, 255)

            cv2.rectangle(display, (0, 0), (w, h), color, 3)

            # Info panel
            overlay = display.copy()
            panel_h = 180 + len(results[:3]) * 22
            cv2.rectangle(overlay, (20, 20), (w - 20, panel_h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.82, display, 0.18, 0, display)

            y = 55
            cv2.putText(display, "SECURITY SYSTEM ACTIVE", (35, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            y += 40
            cv2.putText(display, top_gesture.upper(), (35, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)
            y += 38
            cv2.putText(display, action, (35, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            y += 32
            cv2.putText(display, f"Confidence: {top_conf:.1f}%", (35, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 1, cv2.LINE_AA)
            y += 28
            cv2.putText(display, f"Hand detected | Frame #{frame_count}", (35, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

            y += 30
            for i, (g, c) in enumerate(results[:3]):
                cv2.putText(display, f"#{i+1} {g} -- {c:.1f}%", (35, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.48, (190, 190, 190), 1, cv2.LINE_AA)
                y += 22

            cv2.circle(display, (w - 40, 40), 8, (0, 255, 0), -1)

        # FPS counter
        cv2.putText(display, f"{avg_fps:.1f} FPS", (w - 110, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow("Gesture Security System", display)

        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), 27]:
            break
        elif key == ord('s'):
            fname = f"screenshot_{int(time.time())}.png"
            cv2.imwrite(fname, display)
            print(f"💾 Screenshot saved: {fname}")

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Done — {frame_count} frames processed")


# ═══════════════════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_inference_loop()