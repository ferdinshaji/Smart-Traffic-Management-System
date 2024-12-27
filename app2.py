import cv2
from ultralytics import YOLO
import logging
import time

# Suppress logs from ultralytics
logging.getLogger('ultralytics').setLevel(logging.ERROR)

# Load YOLOv8n model
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

# Function to process only the first frame of each video and display vehicle counts for each type
def process_first_frame_of_all_lanes(video_paths):
    for lane_number, video_path in enumerate(video_paths, start=1):
        cap = cv2.VideoCapture(video_path)
        
        # Check if the video stream is opened successfully
        if not cap.isOpened():
            print(f"Error: Could not open video stream for lane {lane_number}")
            continue

        # Read the first frame
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Could not read the first frame of lane {lane_number}")
            cap.release()
            continue
        
        # Run inference on the first frame
        results = model(frame)

        # Initialize dictionary to store vehicle counts
        vehicle_count = { "Car": 0, "Motorcycle": 0, "Bus": 0, "Truck": 0, "Van": 0 }

        # Filter detections for vehicles only and check confidence
        for box in results[0].boxes:
            confidence = box.conf.item()  # Get the confidence of the detection
            class_id = int(box.cls.item())  # Get the class ID as an integer
            if class_id in vehicle_classes and confidence >= confidence_threshold:  # If vehicle and confidence above threshold
                class_name = vehicle_classes[class_id]
                vehicle_count[class_name] += 1

        # Annotate frame with vehicle counts
        cv2.putText(frame, f'Lane {lane_number}: {sum(vehicle_count.values())} vehicles', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw bounding boxes for detected vehicles and display class name with confidence
        for box in results[0].boxes:
            confidence = box.conf.item()
            class_id = int(box.cls.item())
            if class_id in vehicle_classes and confidence >= confidence_threshold:
                # Get box coordinates (xyxy format)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                # Draw the rectangle
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                # Get the class name
                class_name = vehicle_classes[class_id]
                # Display the class name and confidence score
                cv2.putText(frame, f'{class_name} {confidence*100:.2f}%', (int(x1), int(y1) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show the annotated frame for the first frame of the lane
        cv2.imshow(f'Lane {lane_number}', frame)

        # Output the vehicle count for each type of vehicle
        print(f"Lane {lane_number} Vehicle Count: {vehicle_count}")

        # Release the video capture object for the current lane video
        cap.release()

        # Wait briefly before proceeding to the next lane
        cv2.waitKey(1)

    # Close all OpenCV windows after processing all lanes
    cv2.destroyAllWindows()

# Process the first frame of each lane
process_first_frame_of_all_lanes(video_paths)
