"""视频多目标跟踪——YOLO 检测 + ByteTrack 跟踪，输出带 Tracking ID 的标注视频。"""
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLOv8 tracking and save an annotated video.")
    parser.add_argument("--weights", default="weights/best.pt")
    parser.add_argument("--source", default="data/test_video.mp4", help="Input video path.")
    parser.add_argument("--output", default="runs/task2_tracking/track_output.mp4")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="bytetrack.yaml or botsort.yaml")
    parser.add_argument("--conf", type=float, default=0.25)  # 置信度阈值
    return parser.parse_args()


def main():
    args = parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    # persist=True: 帧间维持同一 Tracking ID
    for result in model.track(source=args.source, stream=True, persist=True, tracker=args.tracker, conf=args.conf):
        frame = result.plot()  
        writer.write(frame)
    writer.release()
    print(f"Saved tracked video to {args.output}")


if __name__ == "__main__":
    main()
