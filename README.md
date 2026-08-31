# AI Virtual Painter

A webcam-based virtual painting app built with OpenCV and MediaPipe hand tracking. Use hand gestures to select a color, switch to the eraser, and draw directly in the air on top of the camera feed.

## Features

- Real-time hand tracking through the webcam
- Gesture-based color selection
- Brush and eraser modes
- Smooth drawing on a virtual canvas
- FPS display while the app is running

## How It Works

- The app opens your default camera.
- A toolbar is shown on the left side of the window.
- Two raised fingers are used to select a tool or color from the toolbar.
- One raised index finger is used to draw on the canvas.
- Press `c` to clear the canvas.
- Press `q` to quit the application.

## Requirements

- Python 3.14 or later
- A working webcam
- The following files in the project directory:
  - `app.py`
  - `module.py`
  - `hand_landmarker.task`
  - `yellow.jpg`
  - `blue.jpg`
  - `erase.jpg`
  - `front.jpg`
  - `red.jpg`

## Installation

Create and activate a virtual environment, then install the dependencies:

```bash
pip install opencv-python==4.14.0.94 mediapipe numpy
```

## Run

Start the application with:

```bash
python app.py
```

## Controls

- `c`: clear the canvas
- `q`: quit

## Notes

- The toolbar images are loaded from the project root.
- The hand landmark model file must be present for hand tracking to work.
- The app assumes a 1280x720 camera frame for the drawing area and toolbar layout.

