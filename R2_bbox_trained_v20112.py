from ultralytics import YOLO
import cv2
import os
import json
import time

model = YOLO("runs/detect/train/weights/best.pt")

image_dir = "R2_datta"
image_files = sorted(os.listdir(image_dir))

output_json = "all_frames_result.json"

all_results = []
frame_idx = 0
ts_step = 0.5

for img_name in image_files:
    img_path = os.path.join(image_dir, img_name)
    frame = cv2.imread(img_path)

    if frame is None:
        continue

    start = time.time()
    results = model(frame, verbose=False, iou=0.3, conf=0.45)
    infer_ms = round((time.time() - start) * 1000, 2)

    result = results[0]
    people = []

    if result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)

        for (x1, y1, x2, y2), conf, cid in zip(boxes, confs, cls_ids):
            if result.names[cid] == "person":
                people.append({
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                    "conf": round(float(conf), 3)
                })

    all_results.append({
        "frame": frame_idx,
        "ts": round(frame_idx * ts_step, 1),
        "infer_ms": infer_ms,
        "people": people
    })

    frame_idx += 1

with open(output_json, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

print("✅ JSON만 저장 완료")