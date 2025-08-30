To run the synthetic data generation pipeline, follow these steps:

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the data generation script:
   ```
   blenderproc run synthetic_data_generator.py
   ```

3. (optional) Visualize the generated annotations:
   ```
   python synth_data_gen/generate_label_vis.py
   ```

   4. Convert annotations to YOLO segmentation format:
      Note: This script uses hard-coded paths. Open the file and set the pathes.
   ```
   python pose_estimation/coco2yolo_seg.py 

   ```

   5.Domain randomization for synthetic images (sim2real):
   Note: Also hard-coded paths.
   ```
   python pose_estimation/sim2real_synth_only.py

   ```
    
   6.Split into train/val sets:
   Note: hard-coded ROOT
   ```
   python pose_estimation/split_dataset.py

   ```
   Create a YOLO data YAML.
   
   7.Train a segmentation model on the synthetic dataset:
   (exact command we used)
   ```
  python
  yolo segment train \
  model=yolo11s-seg.pt \
  data=/Users/nivmaman/Documents/cvsa1/data_seg_dr.yaml \
  imgsz=960 \
  epochs=60 \
  batch=8 \
  device=mps \
  workers=4 \
  cache=disk \
  patience=10 \
  lr0=0.01 lrf=0.01 momentum=0.937 weight_decay=0.0005 \
  warmup_epochs=3.0 warmup_momentum=0.8 warmup_bias_lr=0.1 \
  degrees=5.0 translate=0.08 scale=0.20 shear=2.0 perspective=0.0 \
  hsv_h=0.015 hsv_s=0.30 hsv_v=0.30 \
  mosaic=0.4 mixup=0.10 copy_paste=0.4 erasing=0.10 close_mosaic=5 \
  mask_ratio=4

   ```


   8.fine tune:
   ```
  python 
  yolo segment train \
  model=runs/segment/train4/weights/best.pt \
  data=/Users/nivmaman/Documents/cvsa1/data_seg_dr.yaml \
  imgsz=960 \
  epochs=40 \
  batch=8 device=mps workers=4 cache=disk \
  patience=10 \
  lr0=0.002 lrf=0.01 momentum=0.937 weight_decay=0.0005 \
  warmup_epochs=1.0 warmup_momentum=0.9 warmup_bias_lr=0.05 \
  mosaic=0.2 mixup=0.05 copy_paste=0.2 erasing=0.05 close_mosaic=0 \
  degrees=3.0 translate=0.06 scale=0.15 shear=1.0 \
  hsv_h=0.01 hsv_s=0.25 hsv_v=0.25 \
  mask_ratio=4


   ```

  10.refinement- Generate pseudo-labels from real surgical video:
```
   python pose_estimation/make_pseudo_dataset.py \
  --video "/video/4_2_24_A_1.mp4" \
  --model "/ABS/PATH/TO/runs/segment/train4/weights/best.pt" \
  --out_root "/PATH/TO/YOUR/OUT/FOLDER" \
  --fps 5 --conf 0.7 --iou 0.6 --max_det 3 --imgsz 1024 \
  --min_area_ratio 0.0005 --min_solidity 0.8 --max_thinness 4 \
  --topk 1 --debug
```
 11.refinement- Train the refined model:
  Create a mixed data YAML
```
  yolo segment train \
  model=/ABS/PATH/TO/runs/segment/train4/weights/best.pt \
  data=/ABS/PATH/TO/configs/data_seg_mixed.yaml \
  imgsz=960 \
  epochs=25 \
  batch=8 device=mps workers=4 cache=disk \
  patience=10 \
  lr0=0.002 lrf=0.01 momentum=0.937 weight_decay=0.0005 \
  warmup_epochs=1.0 warmup_momentum=0.9 warmup_bias_lr=0.05 \
  mosaic=0.2 mixup=0.05 copy_paste=0.2 erasing=0.05 close_mosaic=0 \
  degrees=3.0 translate=0.06 scale=0.15 shear=1.0 \
  hsv_h=0.01 hsv_s=0.25 hsv_v=0.25
```
12. Pose overlay from segmentation
Single image → predict.py
```
python predict.py \
  --image "/path/to/image.jpg" \
  --model "/ABS/PATH/TO/WEIGHTS/best.pt" \
  --out   "/path/to/out.jpg" \
  --conf  0.7
```
Video → video.py
```
python video.py \
  --video "/Users/nivmaman/Documents/cvsa/video/4_2_24_A_1.mp4" \
  --model "/ABS/PATH/TO/WEIGHTS/best.pt" \
  --out   "/ABS/PATH/TO/results_synthetic_only.mp4" \
  --conf  0.7
  ```
(We actually used pose_from_seg.py, which was provided inside the pose estimation folder, but changed it so it would fit the project's requirements)
   

