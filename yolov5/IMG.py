import torch

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Set model to evaluation mode
model.eval()

# Input image path
image_path = 'shadow.png'  # change to your image path

# Perform detection
results = model(image_path)

# Print and display results
results.print()
results.show()  # opens image in new window
results.save(save_dir='outputs/image')  # saves result to folder
