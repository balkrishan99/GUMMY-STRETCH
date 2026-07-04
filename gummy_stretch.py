"""
gummy_stretch.py
-----------------
GUMMY STRETCH -- real-time webcam "Gum-Gum Fruit" rubber-limb effect.

Show two hands to the camera. Pinch your thumb and a fingertip on one hand
(the "grabber") near a fingertip of your other hand (the "limb"), then drag
away: that finger stretches like rubber, tracked live, and photographically
deformed via an original mesh-warping engine. Let go and it snaps back with
an elastic overshoot.

Built entirely from scratch: MediaPipe Hand Landmarker (Tasks API, 21
landmarks/hand) for tracking, an original radial-basis + relaxation mesh
deformer for the stretch, and hand-rolled spring/easing math for the snap.
No model training. No GPU required.

Controls:
    q / ESC    quit
    SPACE      mirror on/off
    g          toggle hand guide overlay (skeleton + pinch markers)
    w          toggle mesh wireframe debug view
    r          swap which hand is the limb vs. the grabber
    f          toggle photographic mesh warp vs. simple tube renderer

Run:
    python gummy_stretch.py                 # live webcam
    python gummy_stretch.py --selftest      # headless math/state check
"""

import argparse
import os
import sys

import cv2
import numpy as np

from springs import distance
from hand_tracker import (
    make_hand_landmarker, find_pinch,
    split_by_handedness, FINGER_TIPS, TIP_TO_KNUCKLE,
)
from grab_state import GrabState, IDLE, STRETCHING, SNAPPING
from limb_renderer import draw_rubber_limb, draw_anchor_marker, LimbStyle
from finger_mask import build_finger_capsule, tighten_to_skin, estimate_finger_width
from rubber_mesh import RubberMesh, build_region_mesh, warp_mesh_into


HAND_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "models", "hand_landmarker.task")

HAND_BONES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


class AppConfig:
    cam_index = 0
    cam_width = 1280
    cam_height = 720

    pinch_engage_px = 36.0
    pinch_release_px = 58.0
    pinch_smoothing = 0.5
    snap_seconds = 0.32
    snap_punch = 1.6
    attach_radius_px = 55.0

    limb_hand = "Left"
    grabber_hand = "Right"

    mesh_relax_iterations = 3
    mesh_anchor_radius_factor = 0.9   # * finger width
    use_photoreal_warp = True
    use_skin_tighten = True


def draw_hand_skeleton(frame, hand_px, color, hide_index=None):
    pts = [(int(x), int(y)) for x, y in hand_px]
    for a, b in HAND_BONES:
        if hide_index is not None and hide_index in (a, b):
            continue
        cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)
    for i, p in enumerate(pts):
        if i == hide_index:
            continue
        dot_color = (0, 0, 255) if i in (4,) + FINGER_TIPS else (0, 255, 255)
        cv2.circle(frame, p, 4, dot_color, -1, cv2.LINE_AA)


def run(cfg=AppConfig):
    # Imported here (not at module level) so `--selftest` works in
    # environments without mediapipe installed -- it only needs numpy/cv2.
    import mediapipe as mp

    if not os.path.exists(HAND_MODEL_PATH):
        print(
            f"Missing hand model at {HAND_MODEL_PATH}\n"
            "Download it with:\n"
            "  mkdir -p models\n"
            "  curl -sSL -o models/hand_landmarker.task "
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/latest/hand_landmarker.task",
            file=sys.stderr,
        )
        return 1

    cap = cv2.VideoCapture(cfg.cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.cam_height)
    if not cap.isOpened():
        print("Could not open webcam. Check camera permissions/index.", file=sys.stderr)
        return 1

    landmarker = make_hand_landmarker(HAND_MODEL_PATH, max_hands=2)
    grab = GrabState(
        engage_px=cfg.pinch_engage_px, release_px=cfg.pinch_release_px,
        smoothing=cfg.pinch_smoothing, snap_seconds=cfg.snap_seconds,
        snap_punch=cfg.snap_punch, attach_radius=cfg.attach_radius_px,
    )

    mirror = True
    show_guides = True
    show_wireframe = False
    swap_roles = False
    use_warp = cfg.use_photoreal_warp
    mesh_cache = None
    frame_index = 0

    print("GUMMY STRETCH running. q=quit  SPACE=mirror  g=guides  "
          "w=wireframe  r=swap hands  f=toggle photoreal warp")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if mirror:
            frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(frame_index * 1000.0 / 30.0)
        frame_index += 1
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        limb_label = cfg.grabber_hand if swap_roles else cfg.limb_hand
        grabber_label = cfg.limb_hand if swap_roles else cfg.grabber_hand
        hands_by_label = split_by_handedness(
            result.hand_landmarks, getattr(result, "handedness", None), w, h
        )
        limb_px = hands_by_label.get(limb_label)
        grabber_px = hands_by_label.get(grabber_label)

        grabber_pinch_point, grabber_pinch_dist = None, float("inf")
        if grabber_px is not None:
            grabber_pinch_point, grabber_pinch_dist, _ = find_pinch(grabber_px)

        limb_fingertips = None
        if limb_px is not None:
            limb_fingertips = {idx: limb_px[idx] for idx in FINGER_TIPS}

        grab.update(grabber_pinch_point, grabber_pinch_dist, limb_fingertips)
        if grab.state == IDLE:
            mesh_cache = None

        if grab.is_active() and limb_px is not None and grab.grabbed_finger is not None:
            tip_idx = grab.grabbed_finger
            knuckle_idx = TIP_TO_KNUCKLE[tip_idx]
            live_knuckle = limb_px[knuckle_idx]
            pull_point = grab.current_pull_point(limb_fingertips)
            finger_w = estimate_finger_width(limb_px)

            if use_warp:
                if mesh_cache is None or mesh_cache["finger"] != tip_idx:
                    capsule = build_finger_capsule(h, w, limb_px, tip_idx, finger_w)
                    region = (tighten_to_skin(frame, capsule) if cfg.use_skin_tighten
                              else capsule)
                    built = build_region_mesh(region)
                    mesh_cache = None
                    if built is not None:
                        verts, tris = built
                        dists = np.linalg.norm(verts - np.array(live_knuckle), axis=1)
                        radius = finger_w * cfg.mesh_anchor_radius_factor
                        anchors = list(np.where(dists < radius)[0])
                        if not anchors:
                            anchors = [int(dists.argmin())]
                        rubber = RubberMesh(verts, tris, anchors,
                                            relax_iters=cfg.mesh_relax_iterations)
                        texture = np.dstack([frame, region]).copy()
                        mesh_cache = {
                            "finger": tip_idx, "rubber": rubber, "texture": texture,
                            "verts": verts, "tris": tris, "knuckle0": live_knuckle,
                        }

                if mesh_cache is not None and pull_point is not None:
                    mc = mesh_cache
                    shift_x = live_knuckle[0] - mc["knuckle0"][0]
                    shift_y = live_knuckle[1] - mc["knuckle0"][1]
                    anchor_now = (mc["knuckle0"][0] + shift_x, mc["knuckle0"][1] + shift_y)
                    reach = distance(mc["knuckle0"], limb_px[tip_idx])
                    deformed = mc["rubber"].deform(anchor_now, pull_point, reach)
                    deformed[:, 0] += shift_x
                    deformed[:, 1] += shift_y
                    warp_mesh_into(mc["texture"], mc["verts"], deformed, mc["tris"], frame)
                    if show_wireframe:
                        for tri in mc["tris"]:
                            cv2.polylines(frame, [deformed[tri].astype(np.int32)],
                                          True, (0, 255, 255), 1, cv2.LINE_AA)
            elif pull_point is not None:
                draw_rubber_limb(frame, live_knuckle, pull_point)
                draw_anchor_marker(frame, live_knuckle)

        if show_guides:
            if limb_px is not None:
                hide = (grab.grabbed_finger
                        if grab.is_active() and use_warp else None)
                draw_hand_skeleton(frame, limb_px, (0, 200, 0), hide_index=hide)
            if grabber_px is not None:
                draw_hand_skeleton(frame, grabber_px, (0, 170, 255))
            if grabber_pinch_point is not None:
                color = (0, 255, 0) if grab.latch.active else (0, 165, 255)
                cv2.circle(frame, (int(grabber_pinch_point[0]), int(grabber_pinch_point[1])),
                           8, color, 2, cv2.LINE_AA)

        cv2.imshow("GUMMY STRETCH", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break
        elif key == ord(' '):
            mirror = not mirror
        elif key == ord('g'):
            show_guides = not show_guides
        elif key == ord('w'):
            show_wireframe = not show_wireframe
        elif key == ord('r'):
            swap_roles = not swap_roles
        elif key == ord('f'):
            use_warp = not use_warp
            mesh_cache = None

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()
    return 0


def selftest():
    print("Running GUMMY STRETCH self-test (no camera, no mediapipe needed)...")

    from springs import overshoot_curve, radial_falloff, Hysteresis, distance

    assert abs(overshoot_curve(0.0) - 0.0) < 1e-9
    assert abs(overshoot_curve(1.0) - 1.0) < 1e-9
    peak = max(overshoot_curve(t / 200) for t in range(201))
    assert peak > 1.0, "overshoot_curve should rise past 1.0 before settling"
    print(f"  overshoot_curve: endpoints OK, peak = {peak:.3f}")

    assert radial_falloff(0.0, 100.0) == 1.0
    assert radial_falloff(150.0, 100.0) == 0.0
    mid = radial_falloff(50.0, 100.0)
    assert 0.0 < mid < 1.0
    print(f"  radial_falloff: endpoints OK, midpoint = {mid:.3f}")

    h = Hysteresis(engage_below=40.0, release_above=60.0)
    assert h.update(100.0) is False
    assert h.update(30.0) is True
    assert h.update(50.0) is True       # dead band: stays latched
    assert h.update(70.0) is False
    print("  Hysteresis: dead-band latch OK")

    clock = {"t": 0.0}
    grab = GrabState(engage_px=36.0, release_px=58.0, snap_seconds=0.3,
                      clock=lambda: clock["t"])
    grab.update(None, float("inf"), None)
    assert grab.state == IDLE
    grab.update((200, 200), 10.0, {8: (205, 205)})
    assert grab.state == STRETCHING, grab.state
    grab.update((400, 300), 12.0, {8: (205, 205)})
    assert grab.state == STRETCHING
    grab.update((400, 300), 100.0, {8: (205, 205)})
    assert grab.state == SNAPPING, grab.state
    clock["t"] = 0.31
    grab.update(None, float("inf"), None)
    assert grab.state == IDLE, grab.state
    print("  GrabState: idle->stretching->snapping->idle OK")

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    centerline = draw_rubber_limb(frame, (100, 240), (520, 200))
    assert len(centerline) == LimbStyle.samples + 1
    assert frame.sum() > 0
    print(f"  draw_rubber_limb: drew {len(centerline)} samples, frame non-empty")

    short_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    long_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    draw_rubber_limb(short_frame, (300, 240), (360, 240))
    draw_rubber_limb(long_frame, (60, 240), (600, 240))
    short_density = short_frame.sum() / 60.0
    long_density = long_frame.sum() / 540.0
    assert long_density < short_density, "longer limb should be thinner per unit length"
    print(f"  volume conservation: short {short_density:.0f} > long {long_density:.0f} per px")

    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(mask, (100, 100), 60, 255, -1)
    built = build_region_mesh(mask, boundary_points=20)
    assert built is not None
    verts, tris = built
    assert len(verts) >= 4 and len(tris) >= 1
    print(f"  build_region_mesh: {len(verts)} verts, {len(tris)} tris")

    rubber = RubberMesh(verts, tris, anchor_indices=[int(np.argmin(verts[:, 0]))])
    anchor_pt = tuple(verts[int(np.argmin(verts[:, 0]))])
    far_pt = tuple(verts[int(np.argmax(verts[:, 0]))])
    reach = ((far_pt[0] - anchor_pt[0]) ** 2 + (far_pt[1] - anchor_pt[1]) ** 2) ** 0.5
    deformed = rubber.deform(anchor_point=anchor_pt,
                              pull_point=(far_pt[0] + 40, far_pt[1] - 10),
                              reach=reach)
    assert deformed.shape == verts.shape
    moved = np.linalg.norm(deformed - verts, axis=1)
    assert moved.max() > 0.5, "mesh should visibly deform under a pull"
    # the vertex farthest from the anchor (closest to the original pull
    # target) should move noticeably more than the anchor-adjacent ones
    far_idx = int(np.argmax(verts[:, 0]))
    near_idx = int(np.argmin(verts[:, 0]))
    assert moved[far_idx] > moved[near_idx], \
        "far end of the mesh should move more than the anchored end"
    print(f"  RubberMesh.deform: max displacement = {moved.max():.1f}px "
          f"(far end moved {moved[far_idx]:.1f}px vs anchor end {moved[near_idx]:.1f}px)")

    print("\nself-test PASSED")
    return 0


def main():
    parser = argparse.ArgumentParser(description="GUMMY STRETCH - webcam rubber-limb effect")
    parser.add_argument("--selftest", action="store_true",
                         help="run headless math/state checks, no camera or mediapipe needed")
    args = parser.parse_args()
    if args.selftest:
        return selftest()
    return run()


if __name__ == "__main__":
    sys.exit(main())
