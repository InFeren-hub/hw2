"""从跟踪视频中截取连续 N 帧，用于遮挡/ID 跳变分析。"""
import argparse
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Extract 3-4 consecutive frames for occlusion/ID-switch analysis.")
    parser.add_argument("--video", default="runs/task2_tracking/track_output.mp4")
    parser.add_argument("--start-frame", type=int, required=True)  
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument("--out-dir", default="report/figures/task2_occlusion")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame) 
    for i in range(args.num_frames):
        ok, frame = cap.read()
        if not ok:
            break
        path = out_dir / f"frame_{args.start_frame + i:06d}.jpg"
        cv2.imwrite(str(path), frame)
        print(path)
    cap.release()


if __name__ == "__main__":
    main()
