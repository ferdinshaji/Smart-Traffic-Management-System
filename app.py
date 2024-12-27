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

# Define a list of video paths for each lane
video_paths = [
    'videos/traffic video1.mp4',  # Video for Lane 1
    'videos/traffic video2.mp4',  # Video for Lane 2
    'videos/traffic video3.mp4',  # Video for Lane 3
    'videos/traffic video4.mp4'   # Video for Lane 4
]

# Define the time interval in seconds (e.g., process a frame every 2 seconds)
n_seconds = 2

# Function to process each video and display the output
def process_video(video_path, lane_number):
    cap = cv2.VideoCapture(video_path)
    
    # Check if the video stream is opened successfully
    if not cap.isOpened():
        return

    prev_time = time.time()  # Store the time of the last processed frame

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        current_time = time.time()

        # Process only if n seconds have passed
        if current_time - prev_time >= n_seconds:
            # Run inference on the frame with higher resolution for better accuracy
            results = model(frame, imgsz=640)  # Use 640x640 for better detection

            # Filter detections for vehicles only and check confidence
            filtered_boxes = []
            for box in results[0].boxes:
                confidence = box.conf.item()  # Get the confidence of the detection
                class_id = box.cls.item()  # Get the class ID
                if class_id in vehicle_classes and confidence >= confidence_threshold:  # If vehicle and confidence above threshold
                    filtered_boxes.append((box, confidence, class_id))

            # Count of detected vehicles
            lane_vehicle_count = len(filtered_boxes)
            
            # Annotate frame with vehicle count and bounding boxes
            cv2.putText(frame, f'Lane {lane_number}: {lane_vehicle_count} vehicles', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw bounding boxes for detected vehicles and display class name with confidence
            for box, confidence, class_id in filtered_boxes:
                # Get box coordinates (xyxy format)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                # Draw the rectangle
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                # Get the class name
                class_name = vehicle_classes[class_id]
                # Display the class name and confidence score
                cv2.putText(frame, f'{class_name} {confidence*100:.2f}%', (int(x1), int(y1) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Show the annotated frame
            cv2.imshow(f'Lane {lane_number}', frame)

            prev_time = current_time  # Update the time of the last processed frame

        # Exit the loop if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close windows
    cap.release()
    cv2.destroyAllWindows()

# Process all lanes
for i, video_path in enumerate(video_paths, start=1):
    process_video(video_path, i)
