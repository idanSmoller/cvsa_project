import json, os
from pathlib import Path
import cv2
import numpy as np
from pycocotools import mask as maskUtils

# === DATA PATHS ===
DATA_ROOT = Path("/Users/nivmaman/Documents/cvsa1/synthetic_data")
COCO_JSON = DATA_ROOT / "coco_annotations.json"
IMG_DIR   = DATA_ROOT / "images"
OUT_LABELS= DATA_ROOT / "labels"

os.makedirs(OUT_LABELS, exist_ok=True)

with open(COCO_JSON, "r") as f:
    coco = json.load(f)

id2file = {img["id"]: img["file_name"] for img in coco["images"]}
id2size = {img["id"]: (img["width"], img["height"]) for img in coco["images"]}
catmap  = {cat["id"]: 0 for cat in coco["categories"]}  # single class

im2anns = {img["id"]: [] for img in coco["images"]}
for ann in coco["annotations"]:
    im2anns[ann["image_id"]].append(ann)

n_imgs, n_written, n_polys = 0,0,0

for im_id, anns in im2anns.items():
    fname = id2file[im_id]
    w,h   = id2size[im_id]
    stem  = Path(fname).stem
    out_path = OUT_LABELS / f"{stem}.txt"

    lines=[]
    for ann in anns:
        if ann.get("iscrowd",0): continue
        cls = catmap[ann["category_id"]]
        segm = ann["segmentation"]

        # Polygon
        if isinstance(segm, list):
            for seg in segm:
                if len(seg)<6: continue
                xs = seg[0::2]; ys = seg[1::2]
                norm=[]
                for x,y in zip(xs,ys):
                    norm.append(float(x)/w)
                    norm.append(float(y)/h)
                if len(norm)>=6:
                    line = str(cls)+" "+" ".join(f"{v:.6f}" for v in norm)
                    lines.append(line); n_polys+=1

        # RLE
        elif isinstance(segm, dict) and "counts" in segm:
            rle = maskUtils.frPyObjects(segm, h, w)
            m   = maskUtils.decode(rle)
            if len(m.shape)==3: m = m[:,:,0]
            contours,_ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                if len(c)<3: continue
                poly=[]
                for pt in c:
                    x,y = pt[0]
                    poly.append(x/w); poly.append(y/h)
                if len(poly)>=6:
                    line=str(cls)+" "+" ".join(f"{v:.6f}" for v in poly)
                    lines.append(line); n_polys+=1

    with open(out_path,"w") as f:
        f.write("\n".join(lines))

    n_imgs+=1
    if lines: n_written+=1

print(f"✅ Converted {n_imgs} images. Wrote polygons for {n_written} images, total polygons: {n_polys}")
print("Labels saved to:", OUT_LABELS)
