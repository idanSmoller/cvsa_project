# predict.py
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
    rect = cv2.minAreaRect(cnt)                   # (center, (w,h), angle)
    box = cv2.boxPoints(rect).astype(int)         # 4 rectangle corners

    # take farthest pair of corners as the major axis (tip/base)
    dmax, tip, base = 0, None, None
    for i in range(4):
        for j in range(i + 1, 4):
            d = np.linalg.norm(box[i] - box[j])
            if d > dmax:
                dmax, tip, base = d, tuple(box[i]), tuple(box[j])
    return tip, base


def draw_pose_on_image(model: YOLO, img: np.ndarray, conf: float = 0.7) -> np.ndarray:
    results = model.predict(img, conf=conf, retina_masks=True, verbose=False)
    vis = img.copy()

    for r in results:
        if getattr(r, "masks", None) is None:
            continue
        masks = (r.masks.data.cpu().numpy() * 255).astype(np.uint8)
        for m in masks:
            tip_base = mask_to_pose(m)
            if tip_base:
                tip, base = tip_base
                cv2.line(vis, tip, base, (0, 255, 0), 2)
                cv2.circle(vis, tip, 5, (0, 0, 255), -1)    # red tip
                cv2.circle(vis, base, 5, (255, 0, 0), -1)   # blue base
    return vis


def main():
    ap = argparse.ArgumentParser(description="Pose-from-segmentation on a single image.")
    ap.add_argument("--image", required=True, help="Path to input image")
    ap.add_argument("--model", required=True, help="Path to YOLO segmentation weights (.pt)")
    ap.add_argument("--out", default=None, help="Path to save output image (default: <image>_pose.jpg)")
    ap.add_argument("--conf", type=float, default=0.7, help="Confidence threshold")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    model = YOLO(args.model)
    vis = draw_pose_on_image(model, img, conf=args.conf)

    out_path = args.out or str(Path(args.image).with_suffix("").as_posix() + "_pose.jpg")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, vis)
    print(f"[OK] Saved pose image to {out_path}")


if __name__ == "__main__":
    main()
