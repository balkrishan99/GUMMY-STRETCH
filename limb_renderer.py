"""
limb_renderer.py
----------------
Draws the stylized stretchy "rubber limb" tube between an anchor point and
a live tip — used both as a lightweight fallback render mode and as the
visual silhouette that the photographic mesh warp (rubber_mesh.py) gets
composited under.

A quadratic bezier with a perpendicular bulge gives the limb a slight,
organic curve rather than a straight rod; width tapers toward the tip and
thins overall as the limb lengthens (a simple volume-conservation cue).

Original implementation for GUMMY STRETCH.
"""

import numpy as np
import cv2

from springs import lerp, distance


def _perpendicular(dx, dy):
    length = (dx * dx + dy * dy) ** 0.5
    if length < 1e-6:
        return (0.0, 0.0)
    return (-dy / length, dx / length)


def _bezier_point(p0, p1, p2, t):
    mt = 1.0 - t
    return (
        mt * mt * p0[0] + 2 * mt * t * p1[0] + t * t * p2[0],
        mt * mt * p0[1] + 2 * mt * t * p1[1] + t * t * p2[1],
    )


class LimbStyle:
    base_width = 32.0
    tip_width_frac = 0.4
    min_width = 4.0
    thinning_length = 480.0     # longer limb -> proportionally thinner
    bulge_frac = 0.15
    bulge_max = 60.0
    samples = 26
    fill_color = (60, 70, 230)        # BGR
    outline_color = (35, 40, 145)
    highlight_color = (190, 195, 255)
    anchor_color = (45, 215, 255)


def draw_rubber_limb(frame, anchor, tip, style=LimbStyle):
    """Render a tapered, bulging rubber tube from anchor to tip directly
    onto `frame` (BGR, modified in place). Returns the sampled centerline."""
    length = distance(anchor, tip)
    dx, dy = tip[0] - anchor[0], tip[1] - anchor[1]
    perp_x, perp_y = _perpendicular(dx, dy)

    bulge = min(length * style.bulge_frac, style.bulge_max)
    midpoint = lerp(anchor, tip, 0.5)
    control = (midpoint[0] + perp_x * bulge, midpoint[1] + perp_y * bulge)

    thinning = 1.0 / (1.0 + length / style.thinning_length)
    width_at_anchor = max(style.base_width * thinning, style.min_width)
    width_at_tip = max(width_at_anchor * style.tip_width_frac, style.min_width)

    n = style.samples
    left_edge, right_edge, centerline = [], [], []
    prev_point = _bezier_point(anchor, control, tip, 0.0)
    for i in range(n + 1):
        t = i / n
        point = _bezier_point(anchor, control, tip, t)
        if i == 0:
            ahead = _bezier_point(anchor, control, tip, 1.0 / n)
            tangent = (ahead[0] - point[0], ahead[1] - point[1])
        else:
            tangent = (point[0] - prev_point[0], point[1] - prev_point[1])
        norm_x, norm_y = _perpendicular(*tangent)
        w = width_at_anchor + (width_at_tip - width_at_anchor) * t
        left_edge.append((point[0] + norm_x * w, point[1] + norm_y * w))
        right_edge.append((point[0] - norm_x * w, point[1] - norm_y * w))
        centerline.append(point)
        prev_point = point

    polygon = np.array(left_edge + right_edge[::-1], dtype=np.int32)
    cv2.fillPoly(frame, [polygon], style.fill_color, lineType=cv2.LINE_AA)
    cv2.polylines(frame, [polygon], True, style.outline_color, 2, cv2.LINE_AA)

    highlight = []
    for i, c in enumerate(centerline):
        t = i / n
        w = (width_at_anchor + (width_at_tip - width_at_anchor) * t) * 0.32
        highlight.append((int(c[0] + perp_x * w), int(c[1] + perp_y * w)))
    if len(highlight) >= 2:
        thickness = max(1, int(width_at_tip * 0.45))
        cv2.polylines(frame, [np.array(highlight, np.int32)], False,
                      style.highlight_color, thickness, cv2.LINE_AA)

    cv2.circle(frame, (int(tip[0]), int(tip[1])), int(width_at_tip),
               style.fill_color, -1, cv2.LINE_AA)
    return centerline


def draw_anchor_marker(frame, point, style=LimbStyle):
    p = (int(point[0]), int(point[1]))
    cv2.circle(frame, p, 7, style.anchor_color, -1, cv2.LINE_AA)
    cv2.circle(frame, p, 7, (0, 0, 0), 1, cv2.LINE_AA)
