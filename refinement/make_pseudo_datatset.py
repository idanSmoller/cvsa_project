import os, glob, argparse, json
import cv2
import numpy as np
from ultralytics import YOLO

# ------------------------ utils ------------------------
def ensure_dir(p: str):
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def metrics_from_poly(poly_xy: np.ndarray, W: int, H: int):
    """
    Compute geometry metrics used to filter tool-like shapes.
    poly_xy: (N,2) float array in *pixel* coords
    """
    if poly_xy is None or poly_xy.size < 6:
        return None
    cnt = poly_xy.reshape(-1, 1, 2).astype(np.float32)
    area = float(cv2.contourArea(cnt))
    if area <= 0:
        return None
    x, y, w, h = cv2.boundingRect(cnt.astype(np.int32))
    aspect = max(w, h) / max(1, min(w, h))
    hull = cv2.convexHull(cnt.astype(np.float32))
    hull_area = float(cv2.contourArea(hull))
    solidity = area / max(1.0, hull_area)
    peri = float(cv2.arcLength(cnt, True))
    thinness = (peri * peri) / (4.0 * np.pi * max(1.0, area))
    area_ratio = area / float(W * H)
    return dict(area_ratio=area_ratio, aspect=aspect, solidity=solidity, thinness=thinness)

def write_yolo_seg_label(label_path: str, rows_norm: list):
    """
    rows_norm: list of (cls:int, poly_norm: (N,2) normalized to [0,1])
    Format per line: class x1 y1 x2 y2 ... (no confidence)
    """
    with open(label_path, "w") as f:
        for cls, poly_norm in rows_norm:
            flat = " ".join(f"{x:.6f} {y:.6f}" for x, y in poly_norm)
            f.write(f"{cls} {flat}\n")

# ------------------ frame extraction -------------------
def extract_frames_from_video(video_path: str, out_dir_images: str, target_fps: int = 5):
    ensure_dir(out_dir_images)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    stride = max(int(round(src_fps / max(1, target_fps))), 1)
    kept = 0
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fno = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if fno % stride == 0:
            cv2.imwrite(os.path.join(out_dir_images, f"frame_{idx:06d}.jpg"), frame)
            kept += 1
            idx += 1
    cap.release()
    print(f"[OK] Extracted {kept} frames -> {out_dir_images}")
    return kept

# ------------------ pseudo labeling --------------------
def pseudo_label_folder(
    model_path: str,
    img_dir: str,
    out_labels: str,
    conf: float = 0.50,
    iou: float = 0.60,
    max_det: int = 5,
    imgsz: int = 1024,
    # geometry filters (looser defaults to avoid 0-kept)
    min_area_ratio: float = 0.0005,
    max_area_ratio: float = 0.40,
    min_aspect: float = 1.6,
    min_solidity: float = 0.80,
    max_thinness: float = 3.0,
    topk: int = 3,
    debug: bool = False,
):
    ensure_dir(out_labels)
    model = YOLO(model_path)
    img_paths = sorted([p for p in glob.glob(os.path.join(img_dir, "*")) if p.lower().endswith((".jpg", ".jpeg", ".png"))])

    stats = dict(total_images=len(img_paths), with_preds=0, kept_labels=0)
    drop_reasons_total = {"none": 0, "bounds": 0, "area": 0, "aspect": 0, "solidity": 0, "thinness": 0}

    for ip in img_paths:
        im = cv2.imread(ip)
        if im is None:
            continue
        H, W = im.shape[:2]

        results = model.predict(
            source=im, conf=conf, iou=iou, max_det=max_det,
            imgsz=imgsz, retina_masks=True, verbose=False
        )

        kept_rows = []
        drop_reasons = {"none": 0, "bounds": 0, "area": 0, "aspect": 0, "solidity": 0, "thinness": 0}

        had_preds = False
        for r in results:
            if r.masks is None:
                drop_reasons["none"] += 1
                continue
            had_preds = True
            polys = r.masks.xy  # list of (N_i, 2) arrays in *pixel* coords
            confs = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.ones(len(polys))
            clss = r.boxes.cls.cpu().numpy().astype(int) if r.boxes is not None else np.zeros(len(polys), int)

            for poly_xy, c, k in zip(polys, confs, clss):
                poly_xy = np.asarray(poly_xy)
                # bounds check
                if poly_xy.size < 6 or np.any(poly_xy[:, 0] < 0) or np.any(poly_xy[:, 0] >= W) or np.any(poly_xy[:, 1] < 0) or np.any(poly_xy[:, 1] >= H):
                    drop_reasons["bounds"] += 1
                    continue
                m = metrics_from_poly(poly_xy, W, H)
                if m is None:
                    drop_reasons["bounds"] += 1
                    continue
                if not (min_area_ratio <= m["area_ratio"] <= max_area_ratio):
                    drop_reasons["area"] += 1; continue
                if m["aspect"] < min_aspect:
                    drop_reasons["aspect"] += 1; continue
                if m["solidity"] < min_solidity:
                    drop_reasons["solidity"] += 1; continue
                if m["thinness"] > max_thinness:
                    drop_reasons["thinness"] += 1; continue
                kept_rows.append((float(c), int(k), poly_xy))

        if had_preds:
            stats["with_preds"] += 1

        # keep top-k by confidence
        kept_rows.sort(key=lambda x: x[0], reverse=True)
        kept_rows = kept_rows[:topk]

        if kept_rows:
            base = os.path.splitext(os.path.basename(ip))[0]
            label_path = os.path.join(out_labels, base + ".txt")
            rows_norm = []
            for confv, cls, poly_xy in kept_rows:
                poly_norm = poly_xy.copy().astype(np.float32)
                poly_norm[:, 0] /= W
                poly_norm[:, 1] /= H
                rows_norm.append((cls, poly_norm))
            write_yolo_seg_label(label_path, rows_norm)
            stats["kept_labels"] += 1
        else:
            # aggregate drop reasons for diagnostics
            for k in drop_reasons_total:
                drop_reasons_total[k] += drop_reasons[k]
            if debug:
                print(f"[DEBUG drop] {os.path.basename(ip)} -> {drop_reasons}")

    # print summary
    print(json.dumps({
        "summary": stats,
        "drops": drop_reasons_total,
        "img_dir": img_dir,
        "labels_dir": out_labels
    }, indent=2))
    print(f"[OK] Wrote filtered labels for {stats['kept_labels']} images -> {out_labels}")
    return stats

# --------------------------- CLI ---------------------------
def main():
    ap = argparse.ArgumentParser(description="Create pseudo-labeled dataset (YOLO-seg polygons) from video or image folder.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", type=str, help="Path to raw real video")
    src.add_argument("--images", type=str, help="Path to folder of images (optional alternative to --video)")
    ap.add_argument("--model", type=str, required=True, help="Path to trained segmentation .pt")
    ap.add_argument("--out_root", type=str, required=True, help="Root of pseudo dataset (will create images/ and labels/)")
    ap.add_argument("--fps", type=int, default=5, help="Sampling FPS when using --video")
    ap.add_argument("--conf", type=float, default=0.50)
    ap.add_argument("--iou", type=float, default=0.60)
    ap.add_argument("--max_det", type=int, default=5)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--min_area_ratio", type=float, default=0.0005)
    ap.add_argument("--max_area_ratio", type=float, default=0.40)
    ap.add_argument("--min_aspect", type=float, default=1.6)
    ap.add_argument("--min_solidity", type=float, default=0.80)
    ap.add_argument("--max_thinness", type=float, default=3.0)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    img_dir = args.images
    if args.video:
        img_dir = os.path.join(args.out_root, "images")
        ensure_dir(img_dir)
        extract_frames_from_video(args.video, img_dir, target_fps=args.fps)

    lab_dir = os.path.join(args.out_root, "labels")
    ensure_dir(lab_dir)

    pseudo_label_folder(
        model_path=args.model,
        img_dir=img_dir,
        out_labels=lab_dir,
        conf=args.conf, iou=args.iou, max_det=args.max_det, imgsz=args.imgsz,
        min_area_ratio=args.min_area_ratio, max_area_ratio=args.max_area_ratio,
        min_aspect=args.min_aspect, min_solidity=args.min_solidity, max_thinness=args.max_thinness,
        topk=args.topk, debug=args.debug
    )

if __name__ == "__main__":
    main()
