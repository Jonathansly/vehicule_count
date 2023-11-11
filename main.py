import cv2
from sort import *
import math
import numpy as np
from ultralytics import YOLO
import cvzone

cap = cv2.VideoCapture('cars2.mp4')
model = YOLO('yolov8n.pt')

classnames = []

with open('classes.txt', 'r') as f:
    classnames = f.read().splitlines()

tracker = Sort(max_age=20)
line = [320, 350, 620, 350]
counter = []

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

while 1:
    ret, frame = cap.read()
    if not ret:
        cap = cv2.VideoCapture('cars2.mp4')
        continue

    detections = np.empty((0, 5))
    result = model(frame, stream=1)

    for info in result:
        boxes = info.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            classind = box.cls[0]

            conf = math.ceil(conf * 100)
            classind = int(classind)
            objdetec = classnames[classind]

            if objdetec == 'car' or objdetec == 'bus' or objdetec == 'truck' and conf > 60:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                new_detections = np.array([[x1, y1, x2, y2, conf]])
                detections = np.vstack((detections, new_detections))

    track_result = tracker.update(detections)
    cv2.line(frame, (line[0], line[1]), (line[2], line[3]), (0, 255, 255), 3)

    for result in track_result:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)

        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2

        cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cvzone.putTextRect(frame,
                           f'id_vec{id}',
                           [x1 - 8, y1 + 12],
                           thickness=2,
                           scale=1)

        if line[0] < cx < line[2] and line[1] - 20 < cy < line[1] + 20:
            if counter.count(id) == 0:
                counter.append(id)

            cv2.line(frame, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 10)

    cvzone.putTextRect(frame, f'Comptage vehicule: {len(counter)}', [250, 35], thickness=2, scale=1.5)

    # Write the frame to the output video
    out.write(frame)

    cv2.imshow('frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything when the job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
