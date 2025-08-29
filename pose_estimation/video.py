# video.py
import cv2
import numpy as np
import argparse
from ultralytics import YOLO
from pathlib import Path


def mask_to_pose(mask: np.ndarray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect).astype(int)

    dmax, tip, base = 0, None, None
    for i in range(4):
        for j in range(i + 1, 4):
            d = np.linalg.norm(box[i] - box[j])
            if d > dmax:
                dmax, tip, base = d, tuple(box[i]), tuple(box[j])
    return tip, base


def draw_pose_on_frame(model: YOLO, frame: np.ndarray, conf: float = 0.7) -> np.ndarray:
    results = model.predict(frame, conf=conf, retina_masks=True, verbose=False)
    vis = frame.copy()

    for r in results:
        if getattr(r, "masks", None) is None:
            continue
        masks = (r.masks.data.cpu().numpy() * 255).astype(np.uint8)
        for m in masks:
            tip_base = mask_to_pose(m)
            if tip_base:
                tip, base = tip_base
                cv2.line(vis, tip, base, (0, 255, 0), 2)
                cv2.circle(vis, tip, 5, (0, 0, 255), -1)
                cv2.circle(vis, base, 5, (255, 0, 0), -1)
    return vis


def main():
    ap = argparse.ArgumentParser(description="Pose-from-segmentation on video.")
    ap.add_argument("--video", required=True, help="Path to input video")
    ap.add_argument("--model", required=True, help="Path to YOLO segmentation weights (.pt)")
    ap.add_argument("--out", default="pose_out.mp4", help="Path to save output video")
    ap.add_argument("--conf", type=float, default=0.7, help="Confidence threshold")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    model = YOLO(args.model)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        vis = draw_pose_on_frame(model, frame, conf=args.conf)
        writer.write(vis)

    cap.release()
    writer.release()
    print(f"[OK] Saved pose video to {args.out}")


if __name__ == "__main__":
    main()
