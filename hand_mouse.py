

import cv2
import mediapipe as mp
import pyautogui
import time
import math
from pathlib import Path

# ============== SETTINGS ==============

MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")

SMOOTHING = 0.30
PINCH_THRESHOLD = 0.045
DRAG_DELAY = 0.40
DOUBLE_CLICK_WINDOW = 0.35
SCROLL_THRESHOLD = 0.008
SCROLL_SENSITIVITY = 2.0

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.01

SCREEN_W, SCREEN_H = pyautogui.size()

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

# ============== MEDIAPIPE ==============

BaseOptions = mp.tasks.BaseOptions
Vision = mp.tasks.vision
HandLandmarker = Vision.HandLandmarker

options = Vision.HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=Vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ============== HELPERS ==============

def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def finger_up(hand, tip, joint):
    return hand[tip].y < hand[joint].y


# ============== STATE ==============

smooth_x = None
smooth_y = None

left_pinch_active = False
left_pinch_start = 0.0
dragging = False

right_pinch_active = False

pending_left_click = None
scroll_previous_y = None
last_timestamp = 0

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not camera.isOpened():
    raise RuntimeError("Could not open webcam.")

print("Two-Hand Mouse started.")
print("RIGHT hand: move cursor")
print("LEFT hand: click, right-click, scroll and drag")
print("Press Q to quit. Move the mouse to a screen corner to trigger PyAutoGUI failsafe.")

try:
    with HandLandmarker.create_from_options(options) as detector:

        while True:
            ok, frame = camera.read()

            if not ok:
                print("Could not read webcam frame.")
                break

            # Mirrored view for natural selfie-style control
            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp = max(
                time.monotonic_ns() // 1_000_000,
                last_timestamp + 1
            )
            last_timestamp = timestamp

            result = detector.detect_for_video(
                image, timestamp
            )

            right_hand = None
            left_hand = None

            # Match each hand with its handedness label
            for i, landmarks in enumerate(result.hand_landmarks):
                label = "Unknown"

                if i < len(result.handedness):
                    categories = result.handedness[i]
                    if categories:
                        label = categories[0].category_name

                if label == "Right":
                    right_hand = landmarks
                elif label == "Left":
                    left_hand = landmarks

                # Draw detected fingertips
                for point in landmarks:
                    px = int(point.x * width)
                    py = int(point.y * height)
                    cv2.circle(
                        frame, (px, py), 3, (0, 220, 0), -1
                    )

                cv2.putText(
                    frame, label,
                    (int(landmarks[0].x * width),
                     int(landmarks[0].y * height) - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 200, 0), 2
                )

            now = time.monotonic()
            status = "Show both hands"

            # ========== RIGHT HAND: CURSOR ==========

            if right_hand is not None:
                index = right_hand[8]

                target_x = int(index.x * SCREEN_W)
                target_y = int(index.y * SCREEN_H)

                if smooth_x is None:
                    smooth_x = target_x
                    smooth_y = target_y
                else:
                    smooth_x += SMOOTHING * (
                        target_x - smooth_x
                    )
                    smooth_y += SMOOTHING * (
                        target_y - smooth_y
                    )

                cursor_x = max(
                    0, min(int(smooth_x), SCREEN_W - 1)
                )
                cursor_y = max(
                    0, min(int(smooth_y), SCREEN_H - 1)
                )

                # Do not move cursor while dragging:
                # the cursor is still moved by the right hand
                pyautogui.moveTo(cursor_x, cursor_y)

                status = "RIGHT HAND: Move cursor"

            # ========== LEFT HAND: GESTURES ==========

            if left_hand is not None:
                thumb = left_hand[4]
                index = left_hand[8]
                middle = left_hand[12]

                index_up = finger_up(left_hand, 8, 6)
                middle_up = finger_up(left_hand, 12, 10)
                ring_up = finger_up(left_hand, 16, 14)
                pinky_up = finger_up(left_hand, 20, 18)

                left_pinch = (
                    distance(thumb, index) < PINCH_THRESHOLD
                )

                right_pinch = (
                    distance(thumb, middle) < PINCH_THRESHOLD
                    and not left_pinch
                )

                scroll_mode = (
                    index_up and middle_up
                    and not ring_up and not pinky_up
                    and not left_pinch and not right_pinch
                )

                # Complete a pending single click only after
                # the double-click window expires.
                if (
                    pending_left_click is not None
                    and now - pending_left_click > DOUBLE_CLICK_WINDOW
                ):
                    pyautogui.click()
                    pending_left_click = None

                # LEFT CLICK / DOUBLE CLICK / DRAG
                if left_pinch:
                    if not left_pinch_active:
                        left_pinch_active = True
                        left_pinch_start = now

                    if (
                        not dragging
                        and now - left_pinch_start >= DRAG_DELAY
                    ):
                        # Avoid leaving an earlier click pending
                        if pending_left_click is not None:
                            pyautogui.click()
                            pending_left_click = None

                        if right_hand is not None:
                            pyautogui.moveTo(cursor_x, cursor_y)

                        pyautogui.mouseDown()
                        dragging = True

                    if dragging:
                        status = "LEFT HAND: DRAG - release pinch to drop"
                    else:
                        status = "LEFT HAND: Pinch for click"

                elif left_pinch_active:
                    if dragging:
                        pyautogui.mouseUp()
                        dragging = False
                        status = "DROP"

                    else:
                        if (
                            pending_left_click is not None
                            and now - pending_left_click
                            <= DOUBLE_CLICK_WINDOW
                        ):
                            pyautogui.doubleClick(interval=0.08)
                            pending_left_click = None
                            status = "DOUBLE CLICK"
                        else:
                            if pending_left_click is not None:
                                pyautogui.click()

                            pending_left_click = now
                            status = "LEFT CLICK"

                    left_pinch_active = False

                # RIGHT CLICK: thumb + middle pinch
                if right_pinch and not dragging:
                    if not right_pinch_active:
                        if pending_left_click is not None:
                            pyautogui.click()
                            pending_left_click = None

                        pyautogui.rightClick()
                        right_pinch_active = True
                        status = "RIGHT CLICK"
                else:
                    right_pinch_active = False

                # SCROLL: left index + middle extended
                if scroll_mode and not dragging:
                    if scroll_previous_y is not None:
                        delta = scroll_previous_y - index.y

                        if abs(delta) > SCROLL_THRESHOLD:
                            amount = int(
                                delta * height * SCROLL_SENSITIVITY
                            )
                            amount = max(-8, min(8, amount))

                            if amount:
                                pyautogui.scroll(amount)

                    scroll_previous_y = index.y
                    status = "SCROLL: move left hand up/down"
                else:
                    scroll_previous_y = None

            else:
                # Release drag if left hand disappears
                if dragging:
                    pyautogui.mouseUp()
                    dragging = False

                # Finish a pending single click
                if pending_left_click is not None:
                    pyautogui.click()
                    pending_left_click = None

                left_pinch_active = False
                right_pinch_active = False
                scroll_previous_y = None

            # Reset cursor smoothing if the right hand disappears
            if right_hand is None:
                smooth_x = None
                smooth_y = None

            cv2.putText(
                frame, status,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (0, 255, 0), 2
            )

            cv2.putText(
                frame, "Q = Quit",
                (10, height - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1
            )

            cv2.imshow("Two-Hand Mouse", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    if dragging:
        pyautogui.mouseUp()

    camera.release()
    cv2.destroyAllWindows()
    print("Two-Hand Mouse stopped.")
