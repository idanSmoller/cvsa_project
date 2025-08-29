import cv2
import numpy as np
import os
import argparse
from ultralytics import YOLO


# ---------- Utility: mask -> tip/base ----------
def mask_to_pose(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)  # (center, (w,h), angle)
    box = cv2.boxPoints(rect).astype(int)

    # get the 2 farthest points = major axis endpoints
    dmax, tip, base = 0, None, None
    for i in range(4):
        for j in range(i+1, 4):
            d = np.linalg.norm(box[i] - box[j])
            if d > dmax:
                dmax, tip, base = d, tuple(box[i]), tuple(box[j])
    return tip, base


# ---------- Single Image ----------
def pose_from_image(model, img_path, out_path, conf=0.5):
    img = cv2.imread(img_path)
    results = model.predict(img, conf=conf, retina_masks=True, verbose=False)

    for r in results:
        if not hasattr(r, "masks") or r.masks is None:
            continue
        masks = r.masks.data.cpu().numpy().astype(np.uint8)
        for m in masks:
            mask = (m * 255).astype(np.uint8)
            tip_base = mask_to_pose(mask)
            if tip_base:
                tip, base = tip_base
                cv2.line(img, tip, base, (0, 255, 0), 2)
                cv2.circle(img, tip, 5, (0, 0, 255), -1)   # red tip
                cv2.circle(img, base, 5, (255, 0, 0), -1) # blue base

    cv2.imwrite(out_path, img)
    print(f"[OK] Saved pose image to {out_path}")


# ---------- Folder of Images ----------
def pose_from_folder(model, folder_path, out_folder, conf=0.5):
    os.makedirs(out_folder, exist_ok=True)
    for fname in os.listdir(folder_path):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            in_path = os.path.join(folder_path, fname)
            out_path = os.path.join(out_folder, fname)
            pose_from_image(model, in_path, out_path, conf)


# ---------- Video ----------
def pose_from_video(model, video_path, out_path, conf=0.5):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame, conf=conf, retina_masks=True, verbose=False)

        for r in results:
            if not hasattr(r, "masks") or r.masks is None:
                continue
            masks = r.masks.data.cpu().numpy().astype(np.uint8)
            for m in masks:
                mask = (m * 255).astype(np.uint8)
                tip_base = mask_to_pose(mask)
                if tip_base:
                    tip, base = tip_base
                    cv2.line(frame, tip, base, (0, 255, 0), 2)
                    cv2.circle(frame, tip, 5, (0, 0, 255), -1)
                    cv2.circle(frame, base, 5, (255, 0, 0), -1)

        out.write(frame)

    cap.release()
    out.release()
    print(f"[OK] Saved pose video to {out_path}")


# ---------- Main ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None, help="Path to single image")
    parser.add_argument("--video", type=str, default=None, help="Path to video")
    parser.add_argument("--folder", type=str, default=None, help="Path to folder of images")
    parser.add_argument("--out", type=str, default="pose_out", help="Output file/folder path")
    parser.add_argument("--conf", type=float, default=0.7, help="Confidence threshold")
    args = parser.parse_args()

    model = YOLO("runs/segment/finetune11n_e402/weights/best.pt")

    if args.image:
        pose_from_image(model, args.image, args.out, args.conf)
    elif args.video:
        pose_from_video(model, args.video, args.out, args.conf)
    elif args.folder:
        pose_from_folder(model, args.folder, args.out, args.conf)
    else:
        print(" Please provide --image, --video, or --folder")
