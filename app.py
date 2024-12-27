from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model.predict(source="test_image.jpg", save=True)
print("Detection successful!")