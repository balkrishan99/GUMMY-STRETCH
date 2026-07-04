"""
grab_state.py
-------------
State machine for a two-hand "grab and stretch" interaction: one hand's
fingertip (the "limb") gets pinched by the other hand (the "grabber") and
pulled. Tracks idle -> stretching -> snapping-back transitions and the
smoothed live pinch position.

Original implementation for GUMMY STRETCH.
"""

import time

from springs import Hysteresis, lerp, overshoot_curve, clamp

IDLE = "idle"
STRETCHING = "stretching"
SNAPPING = "snapping"


class GrabState:
    def __init__(self, engage_px=36.0, release_px=58.0, smoothing=0.5,
                 snap_seconds=0.32, snap_punch=1.6, attach_radius=55.0,
                 clock=time.monotonic):
        self.latch = Hysteresis(engage_px, release_px)
        self.smoothing = smoothing
        self.snap_seconds = snap_seconds
        self.snap_punch = snap_punch
        self.attach_radius = attach_radius
        self.clock = clock

        self.state = IDLE
        self.grabbed_finger = None     # which limb-hand fingertip index is held
        self.live_pinch = None         # smoothed grabber pinch point
        self.snap_origin = None        # where the recoil starts from
        self.snap_started_at = 0.0

    def _nearest_grabbable(self, point, limb_fingertips):
        """limb_fingertips: dict {finger_index: (x, y)}. Returns nearest
        index within attach_radius, or None."""
        best_idx, best_dist = None, self.attach_radius
        for idx, pt in limb_fingertips.items():
            d = ((pt[0] - point[0]) ** 2 + (pt[1] - point[1]) ** 2) ** 0.5
            if d < best_dist:
                best_idx, best_dist = idx, d
        return best_idx

    def update(self, grabber_pinch_point, grabber_pinch_dist, limb_fingertips):
        """
        grabber_pinch_point : (x, y) or None
        grabber_pinch_dist  : float (thumb<->finger distance), inf if no hand
        limb_fingertips     : {finger_index: (x, y)} for the hand being grabbed,
                               or None if that hand isn't visible
        """
        now = self.clock()
        has_grabber = grabber_pinch_point is not None
        is_pinched = self.latch.update(grabber_pinch_dist) if has_grabber else False

        if has_grabber:
            if self.live_pinch is None:
                self.live_pinch = grabber_pinch_point
            else:
                s = self.smoothing
                self.live_pinch = (
                    self.live_pinch[0] + (grabber_pinch_point[0] - self.live_pinch[0]) * s,
                    self.live_pinch[1] + (grabber_pinch_point[1] - self.live_pinch[1]) * s,
                )

        if self.state == IDLE:
            if is_pinched and has_grabber and limb_fingertips:
                target = self._nearest_grabbable(grabber_pinch_point, limb_fingertips)
                if target is not None:
                    self.grabbed_finger = target
                    self.live_pinch = grabber_pinch_point
                    self.state = STRETCHING

        elif self.state == STRETCHING:
            if not is_pinched or not has_grabber or not limb_fingertips:
                self.snap_origin = self.live_pinch
                self.snap_started_at = now
                self.state = SNAPPING

        elif self.state == SNAPPING:
            elapsed = now - self.snap_started_at
            if elapsed >= self.snap_seconds:
                self.state = IDLE
                self.grabbed_finger = None
            elif is_pinched and has_grabber and limb_fingertips:
                target = self._nearest_grabbable(grabber_pinch_point, limb_fingertips)
                if target is not None:
                    self.grabbed_finger = target
                    self.live_pinch = grabber_pinch_point
                    self.state = STRETCHING

    def current_pull_point(self, limb_fingertips):
        """Where the stretched limb's tip should currently render, or None."""
        if self.state == STRETCHING:
            return self.live_pinch
        if self.state == SNAPPING:
            t = clamp((self.clock() - self.snap_started_at) / self.snap_seconds, 0.0, 1.0)
            rest_point = (
                limb_fingertips.get(self.grabbed_finger)
                if limb_fingertips and self.grabbed_finger is not None
                else None
            )
            if self.snap_origin is None or rest_point is None:
                return None
            return lerp(self.snap_origin, rest_point, overshoot_curve(t, self.snap_punch))
        return None

    def is_active(self):
        return self.state in (STRETCHING, SNAPPING)
