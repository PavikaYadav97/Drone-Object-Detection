import cv2
from ultralytics import YOLO

# Load YOLOv5 model (YOLOv8 also supports older weights like yolov5s.pt)
model = YOLO('yolov5s.pt')  # It will auto-download if not found

def detect_from_camera():
    cap = cv2.VideoCapture(0)
    print("[INFO] Starting webcam...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Camera not available.")
            break

        results = model.predict(source=frame, conf=0.4, verbose=False)

        for r in results:
            frame = r.plot()

        cv2.imshow("YOLOv5 - Webcam Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Exiting webcam.")
            break

    cap.release()
    cv2.destroyAllWindows()

def detect_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    print(f"[INFO] Opening video: {video_path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, conf=0.4, verbose=False)

        for r in results:
            frame = r.plot()

        cv2.imshow("YOLOv5 - Drone Footage", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Exiting video playback.")
            break

    cap.release()
    cv2.destroyAllWindows()

# ----------- UNCOMMENT ONE OF THESE TO RUN -----------
# detect_from_camera()
detect_from_video("/Users/pavika/Downloads/13701478_1080_1920_30fps.mp4")  # replace with your video path
