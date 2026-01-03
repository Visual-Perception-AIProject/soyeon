from ultralytics import YOLO
import cv2
import os
import json
import time

# 1️⃣ 모델 로드
model = YOLO("runs/detect/train/weights/best.pt")

# 2️⃣ 이미지 폴더
image_dir = "R2_datta"
image_files = sorted(os.listdir(image_dir))

# 3️⃣ 출력 파일 (JSON Lines)
output_file = "results_R2_all.jsonl"

frame_idx = 0

# 4️⃣ 파일 열기 (처음부터 새로 씀)
with open(output_file, "w", encoding="utf-8") as f:
    for img_name in image_files:
        img_path = os.path.join(image_dir, img_name)
        frame = cv2.imread(img_path)

        if frame is None:
            continue

        # 5️⃣ 추론 시간 측정
        start_time = time.perf_counter()
        results = model(frame, verbose=False, iou=0.3, conf=0.45)
        end_time = time.perf_counter()

        infer_ms = (end_time - start_time) * 1000

        result = results[0]
        people = []

        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)

            for (x1, y1, x2, y2), conf, cid in zip(boxes, confs, cls_ids):
                if result.names[cid] != "person":
                    continue
                if y2 < 80:
                    continue

                people.append({
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                    "conf": round(float(conf), 2)
                })

        frame_result = {
            "frame": frame_idx,
            "ts": round(frame_idx * 0.5, 2),
            "infer_ms": round(infer_ms, 2),
            "people": people
        }

        # 6️⃣ 한 줄로 바로 저장
        f.write(json.dumps(frame_result, ensure_ascii=False) + "\n")

        frame_idx += 1

print("✅ JSON Lines 형식으로 한 줄씩 저장 완료")
