import cv2
import time
import joblib  # To load the saved model
import numpy as np
from ultralytics import YOLO
import logging

# Suppress logs from ultralytics
logging.getLogger('ultralytics').setLevel(logging.ERROR)

# Load the trained RandomForestRegressor model
model_rf = joblib.load('traffic_timing_model.pkl')

# Load YOLOv8n model for vehicle detection
model = YOLO('yolov8n.pt')

# List of vehicle class IDs according to the COCO dataset
vehicle_classes = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
    17: "Van"
}

# Confidence threshold for displaying detection
confidence_threshold = 0.70

# Define the paths for each lane video
video_paths = [
    'videos/traffic video1.mp4',  # Video for Lane 1
    'videos/traffic video2.mp4',  # Video for Lane 2
    'videos/traffic video3.mp4',  # Video for Lane 3
    'videos/traffic video4.mp4'   # Video for Lane 4
]

# Function to process the first frame of a video and extract vehicle counts
def get_vehicle_counts(video_path):
    cap = cv2.VideoCapture(video_path)
    
    # Check if the video stream is opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video stream for {video_path}")
        return None

    # Read the first frame
    ret, frame = cap.read()
    if not ret:
        print(f"Error: Could not read the first frame from {video_path}")
        cap.release()
        return None
    
    # Run inference on the first frame
    results = model(frame)

    # Initialize dictionary to store vehicle counts
    vehicle_count = { "Car": 0, "Motorcycle": 0, "Bus": 0, "Truck": 0, "Van": 0 }

    # Filter detections for vehicles and count them
    for box in results[0].boxes:
        confidence = box.conf.item()
        class_id = int(box.cls.item())
        if class_id in vehicle_classes and confidence >= confidence_threshold:
            class_name = vehicle_classes[class_id]
            vehicle_count[class_name] += 1

    # Release the video capture object and return the vehicle counts
    cap.release()
    return vehicle_count

# Function to predict green light duration based on vehicle counts
def predict_green_light_duration(vehicle_count):
    features = np.array([[vehicle_count['Car'], vehicle_count['Motorcycle'], vehicle_count['Bus'],
                          vehicle_count['Truck'], vehicle_count['Van']]])
    predicted_duration = model_rf.predict(features)
    traffic_timing = int(predicted_duration[0] + 1)  # Add 1 second buffer
    return traffic_timing

# Function to process each lane and determine green light duration
def process_all_lanes(video_paths):
    for lane_number, video_path in enumerate(video_paths, start=1):
        print(f"Processing Lane {lane_number}...")

        # Get the vehicle counts for the current lane
        vehicle_count = get_vehicle_counts(video_path)
        if vehicle_count is None:
            continue
        
        # Print vehicle counts for the current lane
        print(f"Lane {lane_number} Vehicle Count: {vehicle_count}")

        # Predict the green light duration
        traffic_timing = predict_green_light_duration(vehicle_count)
        print(f"Predicted Green Light Duration for Lane {lane_number}: {traffic_timing} seconds")

        # Wait for the predicted green light duration before processing the next lane
        time.sleep(traffic_timing)

# Run the lane processing
process_all_lanes(video_paths)
