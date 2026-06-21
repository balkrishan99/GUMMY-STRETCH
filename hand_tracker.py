"""
hand_tracker.py
---------------
Thin wrapper around MediaPipe's Tasks API HandLandmarker, turning raw
21-point hand landmarks into the simple signals GUMMY STRETCH needs: a
pinch point, pinch distance, and which hand (left/right) it belongs to.

Uses the modern mediapipe.tasks.python.vision API (HandLandmarker with a
.task model bundle), not the legacy mp.solutions graphs.
"""

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from springs import distance, lerp

THUMB_TIP = 4
FINGER_TIPS = (8, 12, 16, 20)   # index, middle, ring, pinky
TIP_TO_KNUCKLE = {8: 5, 12: 9, 16: 13, 20: 17}


def make_hand_landmarker(model_path, max_hands=2):
    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=max_hands,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def landmarks_to_px(landmarks, w, h):
    return [(lm.x * w, lm.y * h) for lm in landmarks]


def find_pinch(hand_px, prefer_any_finger=False):
    """
    Given one hand's 21 pixel-space landmarks, return
    (pinch_point, pinch_distance, finger_tip_index).

    By default measures thumb-to-index. If prefer_any_finger is True,
    picks whichever of index/middle/ring/pinky is currently closest to
    the thumb (handy for a looser, more forgiving pinch gesture).
    """
    thumb = hand_px[THUMB_TIP]
    if not prefer_any_finger:
        finger = hand_px[8]
        finger_idx = 8
    else:
        finger_idx = min(FINGER_TIPS, key=lambda i: distance(thumb, hand_px[i]))
        finger = hand_px[finger_idx]
    d = distance(thumb, finger)
    return lerp(thumb, finger, 0.5), d, finger_idx


def split_by_handedness(hand_landmarks_list, handedness_list, w, h):
    """
    Returns {'Left': px_list_or_None, 'Right': px_list_or_None} from raw
    Tasks-API results. Falls back to left-to-right screen order if the
    handedness classifier doesn't cleanly give one of each.
    """
    out = {"Left": None, "Right": None}
    labeled = []
    for i, lms in enumerate(hand_landmarks_list or []):
        label = None
        if handedness_list and i < len(handedness_list) and handedness_list[i]:
            label = handedness_list[i][0].category_name
        labeled.append((label, landmarks_to_px(lms, w, h)))

    for label, px in labeled:
        if label in out and out[label] is None:
            out[label] = px

    # fallback: two hands seen but labels collided -> sort by x position
    if len(labeled) == 2 and (out["Left"] is None or out["Right"] is None):
        sorted_by_x = sorted(labeled, key=lambda kv: kv[1][0][0])
        out["Left"], out["Right"] = sorted_by_x[0][1], sorted_by_x[1][1]

    return out
