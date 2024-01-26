import cv2
import numpy as np

count_line_position = 650
min_width_rect = 80  # min width rectangle
min_height_rect = 80
vehicle_class=[2,3,4,5,6,7,9]
# Load YOLO
net = cv2.dnn.readNet('y_res/yolov3.weights', 'y_res/yolov3.cfg')
with open('y_res/coco.names', 'r') as f:
    classes = f.read().strip().split('\n')

# web camera
cap = cv2.VideoCapture('video.mp4')

def center_handle(x, y, w, h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x + x1
    cy = y + y1
    return cx, cy

detect = []
offset = 6
counter = 0

while True:
    ret, frame1 = cap.read()
    height, width, _ = frame1.shape

    # Convert frame to blob for YOLO
    blob = cv2.dnn.blobFromImage(frame1, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    # Post-process YOLO output
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.25 and class_id in vehicle_class:  # Class ID 2 corresponds to vehicles in COCO dataset
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in indices:
        index = i
        box = boxes[index]
        x, y, w, h = box
        validate_counter = (w >= min_width_rect) and (h >= min_height_rect)

        if not validate_counter:
            continue

        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
        center = center_handle(x, y, w, h)
        detect.append(center)
        cv2.circle(frame1, center, 4, (0, 0, 255), -1)

        for (x, y) in detect:
            if y < (count_line_position + offset) and y > (count_line_position - offset):
                counter += 1
            detect.remove((x, y))

    cv2.line(frame1, (25, count_line_position), (1200, count_line_position), (255, 127, 0), 3)
    cv2.putText(frame1, "VEHICLE COUNTER: " + str(counter), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    cv2.imshow('Video Original', frame1)

    if cv2.waitKey(1) == 13:
        break

cv2.destroyAllWindows()
cap.release()
