import cv2, os, glob, numpy as np
from pathlib import Path
rng = np.random.default_rng(1337)

# === PATHS ===
SRC_IMG = "/Users/nivmaman/Documents/cvsa1/synthetic_data/images"
SRC_LAB = "/Users/nivmaman/Documents/cvsa1/synthetic_data/labels"
DST_IMG = "/Users/nivmaman/Documents/cvsa1/synthetic_data_dr/images"
DST_LAB = "/Users/nivmaman/Documents/cvsa1/synthetic_data_dr/labels"

os.makedirs(DST_IMG, exist_ok=True)
os.makedirs(DST_LAB, exist_ok=True)

# ---------- augment functions ----------
def rand_gamma(img):
    g = float(rng.uniform(0.7, 1.4))
    lut = np.array([((i/255.0)**g)*255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, lut)

def rand_wb(img):
    gains = 1.0 + rng.normal(0, 0.12, size=3)
    return np.clip(img.astype(np.float32) * gains, 0,255).astype(np.uint8)

def vignette(img):
    h,w = img.shape[:2]
    Y,X = np.ogrid[:h,:w]
    cx,cy = w/2,h/2
    r = np.sqrt((X-cx)**2 + (Y-cy)**2)
    v = 0.55 + 0.45*(1 - (r/r.max())**2)
    return np.clip(img.astype(np.float32)*v[...,None],0,255).astype(np.uint8)

def blur(img):
    if rng.random() < 0.5:
        k = int(rng.integers(3,9))|1
        if rng.random() < 0.5:
            return cv2.GaussianBlur(img,(k,k), rng.uniform(0.5,1.6))
        kernel = np.zeros((k,k),np.float32)
        if rng.random()<0.5: kernel[k//2,:]=1.0/k
        else: kernel[:,k//2]=1.0/k
        return cv2.filter2D(img,-1,kernel)
    return img

def jpeg(img):
    q = int(rng.integers(35, 70))
    _,enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    return cv2.imdecode(enc,1)

def glare(img):
    if rng.random()<0.5:
        h,w = img.shape[:2]
        cx,cy = int(rng.uniform(0.3*w,0.7*w)), int(rng.uniform(0.3*h,0.7*h))
        r  = int(rng.uniform(60,180))
        overlay = img.copy()
        cv2.circle(overlay,(cx,cy),r,(255,255,255),-1,cv2.LINE_AA)
        alpha = rng.uniform(0.08,0.15)
        img = cv2.addWeighted(overlay,alpha,img,1-alpha,0)
    return img

def cables(img):
    h,w = img.shape[:2]
    overlay = img.copy()
    for _ in range(int(rng.integers(1,3))):
        pts = []
        x = int(rng.uniform(0,w))
        y = int(rng.uniform(0.4*h,0.9*h))
        for _ in range(int(rng.integers(3,6))):
            x = np.clip(x+int(rng.normal(0,120)),0,w-1)
            y = np.clip(y+int(rng.normal(0,40)),0,h-1)
            pts.append([x,y])
        pts = np.array(pts,np.int32).reshape((-1,1,2))
        color = (
            int(rng.integers(20,60)),
            int(rng.integers(40,140)),
            int(rng.integers(180,255))
        )  # convert to plain Python ints
        thickness = int(rng.integers(2,4))
        cv2.polylines(overlay,[pts],False,color,thickness,cv2.LINE_AA)
    return cv2.addWeighted(overlay,0.7,img,0.3,0)

def gauze(img):
    if rng.random()<0.3:
        h,w=img.shape[:2]
        x1 = int(rng.uniform(0.1*w, 0.6*w))
        y1 = int(rng.uniform(0.4*h, 0.8*h))
        ww = int(rng.uniform(60, 180))
        hh = int(rng.uniform(40, 120))
        x2,y2 = min(w-1,x1+ww), min(h-1,y1+hh)
        patch = img[y1:y2, x1:x2].copy()
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
        L,A,B = cv2.split(patch)
        L = np.clip(L + rng.integers(30,60), 0, 255).astype(np.uint8)
        patch = cv2.cvtColor(cv2.merge([L,A,B]), cv2.COLOR_LAB2BGR)
        img[y1:y2, x1:x2] = patch
    return img

# ---------- main loop ----------
for p in glob.glob(os.path.join(SRC_IMG,"*.jpg"))+glob.glob(os.path.join(SRC_IMG,"*.png")):
    base = os.path.basename(p)
    img = cv2.imread(p)
    if img is None: continue

    img = cv2.resize(img,(1920,1080),interpolation=cv2.INTER_AREA)

    # pipeline
    img = rand_wb(rand_gamma(img))
    img = glare(img)
    img = blur(img)
    img = cables(img)
    img = gauze(img)
    img = vignette(img)
    img = jpeg(img)

    cv2.imwrite(os.path.join(DST_IMG, base.replace(".png",".jpg")), img)

    # copy label 1:1
    lab = os.path.join(SRC_LAB, os.path.splitext(base)[0]+".txt")
    if os.path.exists(lab):
        dst_lab = os.path.join(DST_LAB, os.path.basename(lab))
        with open(lab,"rb") as fi, open(dst_lab,"wb") as fo: fo.write(fi.read())

print("✅ Domain randomized images saved to", DST_IMG)
