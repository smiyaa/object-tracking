import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort

# Load YOLOv8 model
model = YOLO("yolov8s.pt")

# Open video file
cap = cv2.VideoCapture("traffic_3.mp4")

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
COUNT_ZONE_Y = frame_height // 2  # Counting line

# Initialize SORT tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Tracking variables
counted_ids = set()
vehicle_count = 0
previous_centroids = {}

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.5

# Vehicle labels to detect
vehicle_labels = {"car", "truck", "bus", "motorcycle", "van", "trailer", "SUV", "pickup"}

# Display settings
window_width, window_height = 1280, 720

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)  # Run YOLOv8
    detections = []

    for box in results[0].boxes:
        conf = float(box.conf[0])
        if conf < CONFIDENCE_THRESHOLD:
            continue

        label = model.model.names[int(box.cls[0])]
        if label.lower() not in vehicle_labels:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        detections.append([x1, y1, x2, y2, conf])

    detections = np.array(detections) if detections else np.empty((0, 5))
    tracked_objects = tracker.update(detections)

    for x1, y1, x2, y2, track_id in tracked_objects:
        track_id = int(track_id)

        # Draw bounding box and tracking ID
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Calculate centroid
        centroid_x, centroid_y = int((x1 + x2) // 2), int((y1 + y2) // 2)
        cv2.circle(frame, (centroid_x, centroid_y), 4, (255, 255, 0), -1)

        # Counting logic
        if track_id in previous_centroids:
            prev_cy = previous_centroids[track_id][1]
            if prev_cy < COUNT_ZONE_Y <= centroid_y and track_id not in counted_ids:
                vehicle_count += 1
                counted_ids.add(track_id)
        else:
            if centroid_y >= COUNT_ZONE_Y and track_id not in counted_ids:
                vehicle_count += 1
                counted_ids.add(track_id)

        previous_centroids[track_id] = (centroid_x, centroid_y)

    # Draw counting line
    cv2.line(frame, (0, COUNT_ZONE_Y), (frame_width, COUNT_ZONE_Y), (255, 0, 0), 2)

    # Display vehicle count
    cv2.putText(frame, f"Vehicle Count: {vehicle_count}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

    # Resize and show frame
    frame_resized = cv2.resize(frame, (window_width, window_height))
    cv2.imshow("Traffic Vehicle Counter", frame_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



