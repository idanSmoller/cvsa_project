# split_dataset.py
import os, glob, random, shutil, pathlib

random.seed(42)

ROOT = "/Users/nivmaman/Documents/cvsa1"

# use DR dataset
IM = f"{ROOT}/synthetic_data_dr/images"
LB = f"{ROOT}/synthetic_data_dr/labels"

TRI=f"{ROOT}/data_seg_dr/images/train"; TLL=f"{ROOT}/data_seg_dr/labels/train"
VRI=f"{ROOT}/data_seg_dr/images/val";   VLL=f"{ROOT}/data_seg_dr/labels/val"
for d in [TRI,TLL,VRI,VLL]: pathlib.Path(d).mkdir(parents=True, exist_ok=True)

imgs = sorted(glob.glob(f"{IM}/*.*"))
random.shuffle(imgs)
split = int(0.8*len(imgs))
train, val = imgs[:split], imgs[split:]

def cp(imgs, di, dl):
    for p in imgs:
        b = os.path.basename(p); n,_=os.path.splitext(b)
        shutil.copy2(p, f"{di}/{b}")
        lab = f"{LB}/{n}.txt"
        if os.path.exists(lab):
            shutil.copy2(lab, f"{dl}/{n}.txt")

cp(train, TRI, TLL)
cp(val,   VRI, VLL)

print("train:", len(train), "val:", len(val))
