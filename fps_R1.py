from ultralytics import YOLO
import cv2
import os
import time

#model = YOLO("runs/detect/train2/weights/best.pt")
model=YOLO("yolov8l.pt")  # 기본 사전학습 모델 로드
# 🔹 2. 이미지 폴더
image_dir = "R1_datta"
image_files = sorted(os.listdir(image_dir))

fps_list = []

for img_name in image_files:
    img_path = os.path.join(image_dir, img_name)
    frame = cv2.imread(img_path)

    if frame is None:
        continue

    #추론 시작
    start_time = time.time()

    _ = model(frame, verbose=False, iou=0.3, conf=0.45)

    #추론 종료
    end_time = time.time()

    infer_time = end_time - start_time
    fps = 1.0 / infer_time
    fps_list.append(fps)

    print(f"{img_name} | inference time: {infer_time:.4f}s | FPS: {fps:.2f}")

# 🔹 3. 평균 FPS
avg_fps = sum(fps_list) / len(fps_list)
print(f"\n✅ Average FPS: {avg_fps:.2f}")
