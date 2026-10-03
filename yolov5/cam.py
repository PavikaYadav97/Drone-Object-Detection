import torch
import cv2

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model.eval()

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)
assert cap.isOpened(), 'Cannot access camera'

# Real-time detection
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    result_frame = results.render()[0]  # draw predictions

    cv2.imshow('YOLOv5 Live Camera Feed', result_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
