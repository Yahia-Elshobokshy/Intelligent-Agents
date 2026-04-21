"""
encode_faces.py  —  Run ONCE to build encodings.pkl
Uses HOG (CPU-friendly, fast enough for this dataset size)
"""

import face_recognition
import os
import pickle
import cv2

DATASET_PATH = "dataset"
OUTPUT_FILE  = "encodings.pkl"

known_encodings = []
known_names     = []

print("[INFO] Starting encoding…")

for person in sorted(os.listdir(DATASET_PATH)):
    person_path = os.path.join(DATASET_PATH, person)
    if not os.path.isdir(person_path):
        continue

    print(f"[INFO] Processing: {person}")
    encoded_count = 0

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)

        image = cv2.imread(img_path)
        if image is None:
            print(f"  [SKIP] Unreadable: {img_name}")
            continue

        # ── Resize: keep aspect ratio, cap longest side at 640 ──
        h, w = image.shape[:2]
        scale = min(640 / w, 640 / h, 1.0)          # never upscale
        if scale < 1.0:
            image = cv2.resize(image, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # HOG: fast on CPU, good enough for clear face photos
        locations = face_recognition.face_locations(rgb,
                                                    number_of_times_to_upsample=1,
                                                    model="hog")
        if not locations:
            print(f"  [WARN] No face in {img_name}")
            continue

        encodings = face_recognition.face_encodings(rgb, locations)
        for enc in encodings:
            known_encodings.append(enc)
            known_names.append(person)
            encoded_count += 1

    print(f"  → {encoded_count} face(s) encoded")

print(f"\n[INFO] Saving to {OUTPUT_FILE}…")
with open(OUTPUT_FILE, "wb") as f:
    pickle.dump((known_encodings, known_names), f)

print(f"✅ Done! Total faces encoded: {len(known_encodings)}")