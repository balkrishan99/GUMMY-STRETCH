"""
springs.py
----------
Tiny, dependency-free easing/spring math used to drive the snap-back recoil.

Original implementation written for GUMMY STRETCH.
"""

import math


def lerp(a, b, t):
    """Linear interpolate between 2D points a and b."""
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def overshoot_curve(t, punch=1.7):
    """
    Cubic 'back-out' easing: rises past 1.0 then relaxes down to it.
    `punch` controls how far past 1.0 the curve travels before settling —
    this is what gives the snap-back its rubbery, slightly-too-far recoil.
    """
    t = clamp(t, 0.0, 1.0)
    shifted = t - 1.0
    return 1.0 + (punch + 1.0) * (shifted ** 3) + punch * (shifted ** 2)


def damped_wobble(t, decay=3.2, cycles=2.0):
    """
    Under-damped oscillator approaching 1.0: overshoots, swings back a
    little, and settles — used for a snappier, slightly bouncier recoil
    than the pure cubic curve above.
    """
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    envelope = math.exp(-decay * t)
    return 1.0 - envelope * math.cos(cycles * math.pi * t)


def radial_falloff(d, radius):
    """
    Smoothstep falloff: 1.0 at d=0, 0.0 at d>=radius, smooth in between.
    Used to weight how much a mesh point follows a grab vs. stays put.
    """
    if radius <= 1e-6:
        return 0.0
    t = clamp(1.0 - d / radius, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


class Hysteresis:
    """
    A simple two-threshold latch so a noisy signal (like pinch distance)
    doesn't flicker on/off right at the boundary. Engages when the value
    drops below `engage_below`, releases when it rises above `release_above`.
    """

    def __init__(self, engage_below, release_above):
        self.engage_below = engage_below
        self.release_above = release_above
        self.active = False

    def update(self, value):
        if not self.active and value <= self.engage_below:
            self.active = True
        elif self.active and value >= self.release_above:
            self.active = False
        return self.active
