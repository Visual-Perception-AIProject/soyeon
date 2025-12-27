'''
from ultralytics import YOLO
import cv2
import os
import json

# 1️⃣ 모델 로드
model = YOLO("runs/detect/train/weights/best.pt")

# 2️⃣ 이미지 폴더
image_dir = "R2_datta"
image_files = sorted(os.listdir(image_dir))

# 3️⃣ JSON 저장 폴더
json_dir = "json_results"
os.makedirs(json_dir, exist_ok=True)

for img_name in image_files:
    img_path = os.path.join(image_dir, img_name)
    frame = cv2.imread(img_path)

    if frame is None:
        continue

    # 4️⃣ 추론
    results = model(frame, verbose=False,iou=0.3)
    result = results[0]

    detections = []

    if result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)

        for (x1, y1, x2, y2), conf, cid in zip(boxes, confs, cls_ids):
            detections.append({
                "class": result.names[cid],
                "confidence": round(float(conf), 3),
                "bbox": {
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2)
                }
            })

    # 5️⃣ 이미지별 JSON 구성
    image_result = {
        "image": img_name,
        "detections": detections
    }

    # 6️⃣ JSON 파일명 (확장자 제거)
    json_name = os.path.splitext(img_name)[0] + ".json"
    json_path = os.path.join(json_dir, json_name)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(image_result, f, indent=2, ensure_ascii=False)

    # 7️⃣ 시각화
    annotated = result.plot()
    cv2.imshow("YOLOv8 Inference", annotated)

    if cv2.waitKey(1000) == ord('q'):
        break

cv2.destroyAllWindows()
print("✅ 이미지별 JSON 저장 완료")
'''
from ultralytics import YOLO
import cv2
import os
import json

# 1️⃣ 모델 로드
model = YOLO("runs/detect/train/weights/best.pt")

# 2️⃣ 이미지 폴더
image_dir = "R2_datta"
image_files = sorted(os.listdir(image_dir))

# 3️⃣ JSON 저장 폴더
json_dir = "json_results"
os.makedirs(json_dir, exist_ok=True)

# 4️⃣ bbox 이미지 저장 폴더
annot_dir = "annotated_images"
os.makedirs(annot_dir, exist_ok=True)

for img_name in image_files:
    img_path = os.path.join(image_dir, img_name)
    frame = cv2.imread(img_path)

    if frame is None:
        continue

    # 5️⃣ 추론 (IoU 조정)
    results = model(frame, verbose=False, iou=0.3)
    result = results[0]

    detections = []

    if result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)

        for (x1, y1, x2, y2), conf, cid in zip(boxes, confs, cls_ids):
            detections.append({
                "class": result.names[cid],
                "confidence": round(float(conf), 3),
                "bbox": {
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2)
                }
            })

    # 6️⃣ 이미지별 JSON 저장
    image_result = {
        "image": img_name,
        "detections": detections
    }

    json_name = os.path.splitext(img_name)[0] + ".json"
    json_path = os.path.join(json_dir, json_name)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(image_result, f, indent=2, ensure_ascii=False)

    # 7️⃣ bbox 그려진 이미지 생성 & 저장
    annotated = result.plot()
    save_img_path = os.path.join(annot_dir, img_name)
    cv2.imwrite(save_img_path, annotated)

    # 8️⃣ 화면 출력
    cv2.imshow("YOLOv8 Inference", annotated)

    if cv2.waitKey(1000) == ord('q'):
        break

cv2.destroyAllWindows()
print("✅ JSON + bbox 이미지 저장 완료")
