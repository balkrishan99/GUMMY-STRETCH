"""
finger_mask.py
--------------
Builds a binary mask of a single finger from its MediaPipe hand-landmark
bone chain, then optionally tightens that mask to the real skin outline
using local chroma matching — so the photographic mesh warp only ever
moves real finger pixels, not the rectangular landmark capsule around them.

Original implementation for GUMMY STRETCH.
"""

import numpy as np
import cv2

FINGER_BONE_CHAINS = {
    8: (5, 6, 7, 8),       # index:  knuckle -> tip
    12: (9, 10, 11, 12),   # middle
    16: (13, 14, 15, 16),  # ring
    20: (17, 18, 19, 20),  # pinky
    4: (1, 2, 3, 4),       # thumb
}


def estimate_finger_width(hand_px):
    """Rough finger half-width estimate from average inter-knuckle spacing."""
    knuckles = [5, 9, 13, 17]
    gaps = [
        ((hand_px[knuckles[i]][0] - hand_px[knuckles[i + 1]][0]) ** 2 +
         (hand_px[knuckles[i]][1] - hand_px[knuckles[i + 1]][1]) ** 2) ** 0.5
        for i in range(len(knuckles) - 1)
    ]
    return (sum(gaps) / len(gaps)) if gaps else 40.0


def build_finger_capsule(frame_h, frame_w, hand_px, finger_tip_idx,
                          finger_width, taper_strength=0.0, dilate_px=5):
    """
    Paint a capsule-shaped mask around one finger's bone chain. Thickness
    tapers from base to tip, deepening as `taper_strength` increases (use
    this to thin the mask further as the limb is stretched longer).
    """
    mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    chain = FINGER_BONE_CHAINS.get(finger_tip_idx)
    if chain is None:
        return mask

    base_radius = max(3.0, finger_width * 0.5)
    tip_factor = max(0.15, 1.0 - taper_strength)
    points = [(int(hand_px[i][0]), int(hand_px[i][1])) for i in chain]
    n = len(points)
    radii = [base_radius * (1.0 + (tip_factor - 1.0) * (k / max(1, n - 1)))
             for k in range(n)]

    for p, r in zip(points, radii):
        cv2.circle(mask, p, max(1, int(round(r))), 255, -1, cv2.LINE_AA)
    for i in range(n - 1):
        thickness = max(1, int(round(radii[i] + radii[i + 1])))
        cv2.line(mask, points[i], points[i + 1], 255, thickness, cv2.LINE_AA)

    if dilate_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
        mask = cv2.dilate(mask, kernel)
    return mask


def tighten_to_skin(frame_bgr, capsule_mask, cr_tolerance=22, cb_tolerance=22,
                     erode_px=2):
    """
    Shrink a coarse capsule mask down to the actual skin-colored region
    inside it, using the capsule's own eroded core to sample a reference
    skin chroma (YCrCb). Falls back to the original capsule if the
    resulting region looks too small to trust (segmentation likely failed).
    """
    if capsule_mask is None or not capsule_mask.any():
        return capsule_mask

    core = cv2.erode(capsule_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    if not core.any():
        core = capsule_mask

    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    cr_ref = float(np.median(ycrcb[:, :, 1][core > 0]))
    cb_ref = float(np.median(ycrcb[:, :, 2][core > 0]))

    cr_diff = np.abs(ycrcb[:, :, 1].astype(np.int16) - cr_ref)
    cb_diff = np.abs(ycrcb[:, :, 2].astype(np.int16) - cb_ref)
    skin_like = ((cr_diff <= cr_tolerance) & (cb_diff <= cb_tolerance)).astype(np.uint8) * 255

    tightened = cv2.bitwise_and(skin_like, capsule_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    tightened = cv2.morphologyEx(tightened, cv2.MORPH_CLOSE, kernel)
    tightened = cv2.morphologyEx(tightened, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(tightened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return capsule_mask

    result = np.zeros_like(capsule_mask)
    cv2.drawContours(result, [max(contours, key=cv2.contourArea)], -1, 255, -1)
    if erode_px > 0:
        k = erode_px * 2 + 1
        result = cv2.erode(result, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))

    if result.sum() < 0.2 * capsule_mask.sum():
        return capsule_mask  # segmentation likely failed; don't trust it
    return result
