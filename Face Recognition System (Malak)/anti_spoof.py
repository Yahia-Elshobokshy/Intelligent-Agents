"""
anti_spoof.py  --  Liveness detection using repo's AntiSpoofPredict
Patches Detection paths to use absolute paths.
"""

import os
import sys
import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_PATH = os.path.join(SCRIPT_DIR, "Silent-Face-Anti-Spoofing")

# Add repo to path BEFORE importing src
sys.path.insert(0, REPO_PATH)

# Patch Detection.__init__ BEFORE importing AntiSpoofPredict
import src.anti_spoof_predict as _asp_module

def _patched_detection_init(self):
    caffemodel = os.path.join(REPO_PATH, "resources", "detection_model", "Widerface-RetinaFace.caffemodel")
    deploy = os.path.join(REPO_PATH, "resources", "detection_model", "deploy.prototxt")
    self.detector = cv2.dnn.readNetFromCaffe(deploy, caffemodel)
    self.detector_confidence = 0.6

_asp_module.Detection.__init__ = _patched_detection_init

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name

MODEL_DIR = os.path.join(SCRIPT_DIR, "models", "anti_spoof_models")
DEVICE_ID = 0

_model_test = None
_image_cropper = None
_model_files = None

def _init():
    global _model_test, _image_cropper, _model_files
    if _model_test is not None:
        return
    _model_test = AntiSpoofPredict(DEVICE_ID)
    _image_cropper = CropImage()
    _model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pth")]
    if not _model_files:
        raise FileNotFoundError("No .pth models in %s" % MODEL_DIR)
    print("[AntiSpoof] Loaded %d model(s): %s" % (len(_model_files), _model_files))

def check_liveness(frame, face_box=None):
    """
    Check liveness. If face_box is None, uses repo's RetinaFace detector.
    If provided, uses (left, top, right, bottom) from face_recognition.
    """
    _init()
    
    if face_box is None:
        bbox = _model_test.get_bbox(frame)
    else:
        left, top, right, bottom = face_box
        bbox = [left, top, right - left + 1, bottom - top + 1]
    
    if bbox is None or bbox[2] <= 0 or bbox[3] <= 0:
        return False, 0.0, "NoFace"
    
    prediction = np.zeros((1, 3))
    for model_name in _model_files:
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": frame,
            "bbox": bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": scale is not None,
        }
        img = _image_cropper.crop(**param)
        result = _model_test.predict(img, os.path.join(MODEL_DIR, model_name))
        prediction += result
    
    label = np.argmax(prediction)
    value = prediction[0][label] / len(_model_files)
    is_live = (label == 1)
    return is_live, float(value), "LIVE" if is_live else "SPOOF"