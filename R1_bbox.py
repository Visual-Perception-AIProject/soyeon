from ultralytics import YOLO
import cv2
import os
import json

model = YOLO("yolov8l.pt")

image_dir = "R1_datta"
image_files = sorted(os.listdir(image_dir))

delay = 500  # 0.5초

for img_name in image_files:
    img_path = os.path.join(image_dir, img_name)
    frame = cv2.imread(img_path)

    if frame is None:
        continue

    # YOLO 추론
    results = model(frame,classes=[0], verbose=False)
    result = results[0]

    detections = []

    # 교수님 코드 방식 그대로 사용
    if result.boxes is not None:
        boxes   = result.boxes.xyxy.cpu().numpy().astype(int)
        confs   = result.boxes.conf.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)

        for (x1, y1, x2, y2), conf, cid in zip(boxes, confs, cls_ids):
            if result.names[cid] != "person":
                continue

            detections.append({
                "class": "person",
                "confidence": round(float(conf), 3),
                "bbox": {
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2)
                }
            })

    output = {
        "image": img_name,
        "detections": detections
    }

    # JSON 출력
    print(json.dumps(output, indent=2, ensure_ascii=False))

    # 시각화 
    annotated = result.plot(line_width=2)
    cv2.imshow("YOLOv8 Detection", annotated)

    if cv2.waitKey(delay) == ord('q'):
        break

cv2.destroyAllWindows()
