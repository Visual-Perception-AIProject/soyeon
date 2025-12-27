from ultralytics import YOLO
import torch
torch.cuda.empty_cache()

def main():
    model = YOLO("yolov8l.pt")

    model.train(
        data="dataset/data.yaml",
        epochs=150,
        imgsz=960,
        batch=2,
        device=0,
        freeze=20

    )

if __name__ == "__main__":
    main()
