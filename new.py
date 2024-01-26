import cv2
import numpy as np

# Load pre-trained models
vehicle_net = cv2.dnn.readNet('path/to/vehicle-detection-model.xml', 'path/to/vehicle-detection-model.bin')
person_net = cv2.dnn.readNet('path/to/person-detection-model.xml', 'path/to/person-detection-model.bin')

# Set input video file
video_path = 'path/to/your/video/file.mp4'
cap = cv2.VideoCapture(video_path)

# Define the region of interest (ROI)
roi_y_min, roi_y_max = 300, 600
line_position = 450

# Initialize counters
vehicle_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Get the region of interest
    roi = frame[roi_y_min:roi_y_max, :]

    # Vehicle detection
    vehicle_blob = cv2.dnn.blobFromImage(roi, 0.007843, (672, 384), 127.5)
    vehicle_net.setInput(vehicle_blob)
    vehicle_detections = vehicle_net.forward()

    for detection in vehicle_detections[0, 0, :, :]:
        confidence = detection[2]
        if confidence > 0.5:
            x, y, w, h = map(int, detection[3:7] * np.array([roi.shape[1], roi.shape[0], roi.shape[1], roi.shape[0]]))
            cv2.rectangle(frame, (x, y + roi_y_min), (x + w, y + h + roi_y_min), (0, 255, 0), 2)

            # Check if the vehicle crossed the line
            if y + h > line_position and y < line_position:
                vehicle_count += 1

    # Person detection to avoid counting humans
    person_blob = cv2.dnn.blobFromImage(roi, 0.007843, (672, 384), 127.5)
    person_net.setInput(person_blob)
    person_detections = person_net.forward()

    for detection in person_detections[0, 0, :, :]:
        confidence = detection[2]
        if confidence > 0.5:
            x, y, w, h = map(int, detection[3:7] * np.array([roi.shape[1], roi.shape[0], roi.shape[1], roi.shape[0]]))
            cv2.rectangle(frame, (x, y + roi_y_min), (x + w, y + h + roi_y_min), (0, 0, 255), 2)

    # Draw the counting line
    cv2.line(frame, (0, line_position), (frame.shape[1], line_position), (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('Vehicle Counter', frame)

    if cv2.waitKey(30) & 0xFF == 27:  # Press 'Esc' to exit
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Display the vehicle count
print("Total Vehicles:", vehicle_count)
