import cv2
import numpy as np

# YOLO Launch
nn = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = f.read().splitlines()

# Load video
img = cv2.VideoCapture("assets/vidl.mp4")

while True:
    _, image = img.read()
    image = cv2.resize(image, None, fx=0.7, fy=0.7)
    height, width, _ = image.shape

    # Lines
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    low_yellow = np.array([18, 94, 140])
    up_yellow = np.array([48, 255, 255])
    mask = cv2.inRange(hsv, low_yellow, up_yellow)
    edges = cv2.Canny(mask, 75, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, maxLineGap=100)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Object detecting
    blob = cv2.dnn.blobFromImage(image, 1 / 255, (416, 416), (0, 0, 0), True, crop=False)

    nn.setInput(blob)
    op = nn.getLayerNames()
    op = [op[i[0] - 1] for i in nn.getUnconnectedOutLayers()]
    outs = nn.forward(op)

    boxes = []
    confidences = []
    class_ids = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Draw Rectangles
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = str(round(confidences[i], 2))
        #   color = colors [i]
        cv2.rectangle(image, (x, y), (x + w, y + h), (150, 0, 0), 2)
        cv2.putText(image, label + " " + confidence, (x, y + 10), font, 1, (255, 255, 255), 2)

    cv2.imshow('Image', mask)

    key = cv2.waitKey(1)
    if key == 27:
        break
cv2.destroyAllWindows()
