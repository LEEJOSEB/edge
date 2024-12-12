from ultralytics import YOLO

# Load a YOLOv8n PyTorch model
model = YOLO("yolov8n.engine", task = "detect")


# Run inference
results = model("https://ultralytics.com/images/bus.jpg")

print(results)