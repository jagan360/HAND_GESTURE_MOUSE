
import cv2
import mediapipe as mp
import time
from pathlib import Path

model = Path(__file__).with_name("hand_landmarker.task")

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=str(model)
    ),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=2
)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Cannot open webcam")
    raise SystemExit

start = time.monotonic()
last_timestamp = 0

with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
    while True:
        ok, frame = camera.read()
        if not ok:
            print("Cannot read camera")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        timestamp = max(
            int((time.monotonic() - start) * 1000),
            last_timestamp + 1
        )
        last_timestamp = timestamp

        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = detector.detect_for_video(image, timestamp)

        count = len(result.hand_landmarks)
        cv2.putText(
            frame,
            f"Hands detected: {count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        if result.hand_landmarks:
            for hand in result.hand_landmarks:
                for point in hand:
                    x = int(point.x * frame.shape[1])
                    y = int(point.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

        cv2.imshow("Hand Detection Test - Press Q to exit", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()
