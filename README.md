Hand Gesture Mouse Controller

A Python-based project that uses a webcam to detect hand landmarks and serves as the foundation for controlling a computer mouse with hand gestures.

Technologies

- Python
- OpenCV
- MediaPipe
- PyAutoGUI

Features

- Webcam video capture
- Hand landmark detection
- Real-time visual feedback

Requirements

- Python version compatible with the installed MediaPipe release
- Webcam
- OpenCV
- MediaPipe
- PyAutoGUI

How to Run

1. Clone or download this repository.

2. Install the required packages:
   
   "python -m pip install opencv-python mediapipe pyautogui"

3. Download the MediaPipe hand landmark model and place "hand_landmarker.task" beside "hand_mouse.py".

4. Run:
   
   "python hand_mouse.py"

Press Q to close the webcam window.

Future Improvements

- Control the cursor with the index finger
- Add pinch-to-click functionality
- Add gesture-based scrolling

Author:

JAGANNATH