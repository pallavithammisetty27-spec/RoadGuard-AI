from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolo11n.pt")

# Detect objects in road image
results = model("road.jpg", save=True)

print("Detection completed successfully!")