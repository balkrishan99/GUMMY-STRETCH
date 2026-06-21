# 🍬 GUMMY STRETCH

Real-time "Gum-Gum Fruit" rubber-limb effect for your webcam — pinch one
hand's fingertip with your other hand and drag: the finger stretches like
elastic, tracked live, then snaps back with a punchy overshoot when you
let go.

Built from scratch with **MediaPipe Hand Landmarker** (21 landmarks per
hand, modern Tasks API) + **OpenCV** + a hand-rolled 2D mesh deformation
engine. No custom model training. No GPU required.

## How it works

1. **MediaPipe Hand Landmarker** tracks up to two hands per frame, 21
   landmarks each — one hand is the "limb" (the one that stretches), the
   other is the "grabber" (the one doing the pulling).
2. The grabber's thumb-to-fingertip distance feeds a **hysteresis latch**
   (`springs.py`) so the pinch gesture engages and releases cleanly
   without flickering at the threshold.
3. On pinch, the nearest fingertip of the limb hand is selected, a binary
   mask is painted around that finger's bone chain (`finger_mask.py`) and
   optionally tightened to the real skin outline using local YCrCb chroma
   matching.
4. That mask is triangulated into a mesh (`rubber_mesh.py`). Every frame,
   each vertex is displaced toward the live pinch point, weighted by how
   far **along the rest anchor→tip axis** it sits — the knuckle end stays
   rooted, the tip end follows the pull almost fully. A short relaxation
   pass keeps neighboring triangles from shearing apart at high stretch.
5. The deformed mesh triangles are texture-mapped back onto the frame
   with per-triangle affine warps (`cv2.warpAffine`), feathered at the
   silhouette edge so it blends cleanly into the live video.
6. On release, an **overshoot easing curve** animates the limb back to
   its resting fingertip position, overshooting slightly past it before
   settling — the "snap" feel.

All geometry and image warping — no neural rendering, no training,
runs comfortably on CPU.

## Setup

```bash
git clone https://github.com/<your-username>/GUMMY-STRETCH.git
cd GUMMY-STRETCH
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

mkdir -p models
curl -sSL -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

> This project uses MediaPipe's modern **Tasks API**
> (`HandLandmarker` from `mediapipe.tasks.python.vision`), which needs the
> `.task` model bundle downloaded above rather than the older
> `mp.solutions` graphs. `requirements.txt` pins `mediapipe>=0.10.30` to
> match.

## Run

```bash
python3 gummy_stretch.py            # live webcam
python3 gummy_stretch.py --selftest # headless math/state checks, no camera needed
```

## Controls

| Key   | Action |
|-------|--------|
| `q` / `ESC` | quit |
| `SPACE` | mirror on/off |
| `g` | toggle hand guide overlay (skeleton + pinch markers) |
| `w` | toggle mesh wireframe debug view |
| `r` | swap which hand is the limb vs. the grabber |
| `f` | toggle photographic mesh warp vs. simple tube renderer |

## Project structure

```
GUMMY-STRETCH/
├── gummy_stretch.py   # main app: webcam loop, ties everything together
├── springs.py         # easing curves, hysteresis latch, small math helpers
├── grab_state.py       # idle -> stretching -> snapping state machine
├── hand_tracker.py     # MediaPipe Hand Landmarker (Tasks API) wrapper
├── finger_mask.py       # builds + tightens a finger silhouette mask
├── rubber_mesh.py       # mesh triangulation + axial deformation + texture warp
├── limb_renderer.py     # stylized bezier-tube fallback renderer
├── requirements.txt
└── models/              # put hand_landmarker.task here (see Setup)
```

## Tuning the feel

Most of what controls how the effect *feels* lives in `AppConfig` at the
top of `gummy_stretch.py`:

- `pinch_engage_px` / `pinch_release_px` — pinch gesture hysteresis
- `snap_seconds` / `snap_punch` — snap-back duration and overshoot strength
- `attach_radius_px` — how close the grabber pinch must be to a fingertip
  to attach
- `mesh_relax_iterations` — higher = stiffer, less floppy mesh

## Requirements

- Python 3.9–3.12
- A webcam
- CPU only — no GPU needed

## License

MIT — see [LICENSE](LICENSE).

---

Built by Balkrishan Chaamriya 🍬
