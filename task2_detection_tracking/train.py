"""在 VisDrone 上微调 YOLOv8 目标检测模型。"""
import argparse

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 on VisDrone.")
    parser.add_argument("--data", default="task2_detection_tracking/visdrone.yaml")
    parser.add_argument("--model", default="yolov8s.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default="task2_detection")
    parser.add_argument("--name", default="yolov8s_visdrone")
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        project=args.project,
        name=args.name,
        lr0=0.01,         
        lrf=0.01,        
        cos_lr=True,      
        warmup_epochs=3,  
        warmup_momentum=0.8,
        momentum=0.937,
        weight_decay=0.0005,
        mosaic=1.0,       
        hsv_h=0.015,      
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,     
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
    )
  
    metrics = model.val(data=args.data, imgsz=args.imgsz, project=args.project, name=f"{args.name}_val")
    print(f"mAP50: {metrics.box.map50:.4f}, mAP50-95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
