import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def side_of_line(point, a, b):
    """判断点在直线 AB 的哪一侧"""
    return (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])


def parse_point(text):
    x, y = text.split(",")
    return int(x), int(y)


def parse_args():
    parser = argparse.ArgumentParser(description="Track objects and count unique IDs crossing a virtual line.")
    parser.add_argument("--weights", default="weights/best.pt")
    parser.add_argument("--source", default="data/test_video.mp4", help="Input video path.")
    parser.add_argument("--output", default="runs/task2_tracking/line_count.mp4")
    parser.add_argument("--line-a", default="960,100", type=parse_point)  # 起点
    parser.add_argument("--line-b", default="960,980", type=parse_point)  # 终点
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--conf", type=float, default=0.25)
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
    last_side = {}       # 记录每个跟踪ID上一帧所在侧
    counted_ids = set()  # 已计数的 ID 集合

    for result in model.track(source=args.source, stream=True, persist=True, tracker=args.tracker, conf=args.conf):
        frame = result.plot()
        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            ids = boxes.id.cpu().numpy().astype(int)
            xyxy = boxes.xyxy.cpu().numpy()
            for track_id, box in zip(ids, xyxy):
                cx = int((box[0] + box[2]) / 2)  # 框中心 x
                cy = int((box[1] + box[3]) / 2)  # 框中心 y
                now = side_of_line((cx, cy), args.line_a, args.line_b)
                prev = last_side.get(track_id)
                # 两侧符号变号 → 跨越
                if prev is not None and prev * now < 0:
                    counted_ids.add(track_id)
                last_side[track_id] = now
                cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1)

        cv2.line(frame, args.line_a, args.line_b, (0, 0, 255), 2) 
        cv2.putText(frame, f"Count: {len(counted_ids)}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        writer.write(frame)

    writer.release()
    print(f"Saved line-count video to {args.output}; total count={len(counted_ids)}")


if __name__ == "__main__":
    main()
