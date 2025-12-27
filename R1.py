from ultralytics import YOLO
import cv2
import os
import json

model = YOLO("yolov8s.pt")

image_dir = "R1_datta"
image_files = sorted(os.listdir(image_dir))  # 순서 고정

delay = 500  # 0.5초 = 500ms

for img_name in image_files:
    img_path = os.path.join(image_dir, img_name)
    frame = cv2.imread(img_path)

    if frame is None:
        continue

    # YOLO 추론
    results = model(frame, verbose=False)
    result = results[0]

    annotated = result.plot()
    cv2.imshow("YOLOv8 Detection", annotated)

    if cv2.waitKey(delay) == ord('q'):
        break

cv2.destroyAllWindows()
