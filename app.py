import cv2
import numpy as np
from ultralytics import YOLO
from utils.color_detection import get_dominant_color, classify_color

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Class IDs for cars & people (COCO)
CAR_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
PERSON_CLASS = 0

# Open webcam or video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    car_count = 0
    person_count = 0

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if cls in CAR_CLASSES:
            car_img = frame[y1:y2, x1:x2]
            avg_color = get_dominant_color(car_img)
            car_color = classify_color(avg_color)
            car_count += 1

            if car_color == 'blue':
                color = (0, 0, 255)  # red rectangle
            else:
                color = (255, 0, 0)  # blue rectangle

            label = f"Car ({car_color})"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        elif cls == PERSON_CLASS:
            person_count += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Person", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show total counts
    cv2.putText(frame, f"Cars: {car_count}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    cv2.putText(frame, f"People: {person_count}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

    cv2.imshow("Traffic Signal Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
