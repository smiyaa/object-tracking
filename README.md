# Vehicle Detection and Counting using YOLOv8 and SORT

This project utilizes **YOLOv8** for vehicle detection and **SORT (Simple Online and Realtime Tracker)** for tracking vehicles in a traffic video. The system counts vehicles as they cross a designated line.

## Features
- **Real-time vehicle detection** using YOLOv8
- **Tracking** of detected vehicles using SORT
- **Vehicle counting** as they cross a predefined counting zone
- **Display of bounding boxes and tracking IDs**
- **Resizable display window** for better visualization

## Requirements
Make sure you have the following dependencies installed:

```bash
pip install ultralytics opencv-python numpy filterpy
```

Additionally, download the **YOLOv8 model weights** if not available:

```bash
wget https://github.com/ultralytics/assets/releases/download/v8.0.0/yolov8s.pt
```

## Usage
1. Place your traffic video file in the project directory.
2. Update the video file path in the script (`traffic_3.mp4`).
3. Run the script:

```bash
python vehicle_counter.py
```

4. Press `q` to exit the video window.

## Code Explanation
- **YOLOv8 is used** to detect vehicles in each frame.
- **SORT tracker** assigns unique IDs to detected vehicles.
- Vehicles crossing the counting line are counted once.
- The **count is displayed** on the video frame in real-time.

## Customization
- Change the **COUNT_ZONE_Y** variable to adjust the counting line.
- Modify `CONFIDENCE_THRESHOLD` to fine-tune detections.
- Add new vehicle classes to `vehicle_labels` if needed.

## Sample Output
A window displays the processed video with:
- Bounding boxes and tracking IDs for detected vehicles.
- A counting line marking the counting zone.
- A real-time vehicle count displayed at the top.

## License
This project is for educational and research purposes. Modify and use it as needed!

