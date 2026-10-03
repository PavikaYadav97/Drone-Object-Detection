import torch
import cv2

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model.eval()

# Video file path
video_path = 'sample_video.mp4'  # change to your video path
cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), 'Cannot open video'

# Video writer setup
width = int(cap.get(3))
height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter('outputs/output_video.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

# Process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    result_frame = results.render()[0]  # get frame with boxes

    out.write(result_frame)
    cv2.imshow('YOLOv5 Video Detection', result_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
