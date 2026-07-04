# GUMMY STRETCH

Real-time webcam rubber-limb stretching effect built with MediaPipe, OpenCV, and a small custom mesh deformation pipeline.

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

mkdir models
curl -L -o models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
curl -L -o models/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

## Run

```bash
python gummy_stretch.py
python gummy_stretch.py --selftest
```

## Controls

| Key | Action |
| --- | --- |
| q / ESC | Quit |
| SPACE | Toggle mirror mode |
| g | Toggle hand guide overlay |
| w | Toggle mesh wireframe debug view |
| r | Swap limb and grabber hands |
| f | Toggle mesh warp vs. tube renderer |

## Project Structure

- gummy_stretch.py - main app loop
- springs.py - easing and hysteresis helpers
- grab_state.py - stretch/snap state machine
- hand_tracker.py - MediaPipe Hand Landmarker wrapper
- finger_mask.py - finger silhouette mask helpers
- rubber_mesh.py - mesh deformation and warping
- limb_renderer.py - fallback tube renderer
- requirements.txt - Python dependencies
- models/ - downloaded MediaPipe task bundles

## License

MIT
