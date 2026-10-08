# 10th-gen Honda Civic recogniser

A self-taught computer vision project: a model that looks at a car photo and decides whether it is a **10th-gen Honda Civic (2016–2021: FC sedan, FK hatchback, coupe)**. Later I paired it with an object detector so it works on busy street photos.

Built with **Keras 3 on the PyTorch backend**, trained on a laptop GPU (RTX 4050, 6 GB).

![Detect every car, then check each one](images/detect_then_classify.jpg)
*My own street photo: YOLO finds each vehicle, then my Civic model checks each box (green = Civic). Number plates are blurred.*

---

## Results at a glance
| | |
|---|---|
| Best model | EfficientNetV2-S, fine-tuned at 384 × 384, with random-erasing augmentation |
| Accuracy on 43 unseen real-world photos | **97.7% (42/43)**, up from 86% for the first model |
| Street photos | whole photo: Civic missed (40%); detect-then-classify: Civic found (81%) |

The 43-photo test set is small, so treat these numbers as a progress check, not a benchmark.

---

## The story

### 1. Building the dataset (~720 photos)
- Wrote a download script that searches for photos, then filters them automatically:
  - **duplicate removal** with perceptual hashing (catches resized or re-saved copies)
  - an **"is there a car?" check** using a pretrained ConvNeXt-Tiny (ImageNet), so photos of interiors, wheels or logos are dropped
- Then **reviewed every photo by hand** with written rules (e.g. a close-up stays only if 2–3 Civic features are visible together).
- Decided to treat the FC sedan and FK hatchback as one class: from the front they are almost identical, even to people.
- "Not Civic" photos are deliberately similar cars: Honda Accord and City, Toyota Corolla, Mazda 3, older and newer Civic generations.
- Split: 528 training / 99 validation / 97 test. Plus a separate **fresh test set of 43 new photos**, used only to compare models.

### 2. Five model versions, one change at a time
![Accuracy of each model version on the same 43 unseen photos](images/model_evolution.png)

- All models use **transfer learning**: start from a network pretrained on ImageNet, train a new head first (backbone frozen), then fine-tune the last blocks with a small learning rate.
  ![Training curves: stage 1 frozen backbone, stage 2 fine-tuning](images/training_curves_two_stages.png)
- **Adding more photos alone did not help** (86% → 86%).
- **A stronger backbone and higher resolution did** (ConvNeXt-Tiny → EfficientNetV2-B0 → V2-S, 224 → 384 px). At 224 px the small details that separate a Civic from an Accord get squeezed away.
- Fitting EfficientNetV2-S at 384 px on a 6 GB GPU meant batch size 4 and keeping some layers frozen.

### 3. Checking *what* the model looks at (Grad-CAM was misleading)
Grad-CAM on my own street photo pointed at the boot lid centre, not at the 10th-gen's distinctive "claw" tail lights:

![Grad-CAM heatmap pointing at the boot lid](images/gradcam_misleading.png)

So I ran an **occlusion (cover) test**: hide one part at a time with a grey box and watch the score.

![Cover test: hiding the claw lights drops the score, hiding the spoiler does nothing](images/cover_test_tail_lights.png)

- Covering the right claw light: **0.999 → 0.41**. Both lights: **0.095**. Covering the boot centre that Grad-CAM highlighted: **no change**.
- The model **does** rely on the tail lights. Grad-CAM only showed the single strongest spot.
- Because hiding one light could flip the answer, I added **random-erasing augmentation** (random grey boxes during training). That version fixed the side-view mistakes: **93% → 97.7%**.

### 4. Testing on real streets → adding object detection
The test photos mostly show one big car. On my own street photo with several cars, the Civic is small, and the model said **"not Civic"**. Squeezing the whole street into 384 × 384 makes the Civic tiny.

![Whole photo vs YOLO crop](images/whole_photo_vs_yolo_crop.png)

Fix: **detect first, then classify.**
1. A pretrained **YOLOv8n** detector (Ultralytics, COCO) finds every car, truck and bus.
2. Each box is cropped (+10% margin) and resized to 384 × 384, so the Civic model gets the kind of close-up it was trained on.
3. My Civic model scores each crop.

### 5. Exporting to ONNX
To run without PyTorch, Keras or Ultralytics (for a small free server), I exported both models to **ONNX** and rewrote the steps the Ultralytics library normally hides: letterbox resizing, filtering the 8,400 raw YOLO guesses, and non-maximum suppression (keep the most confident box, delete boxes that overlap it).
- Civic model: identical answers on all 43 test photos (largest score difference 0.000002).
- Full pipeline vs the original Python version (44 photos, 64 cars): first try, **3 answers flipped**. The cause: Ultralytics pads photos only up to a multiple of 32 (e.g. 640 × 480), not to a full 640 × 640 square, and resizes with OpenCV. After matching both: **0 flipped, identical boxes and scores**.

---

## Known weaknesses (and what I'd do next)
- **Local cars it never saw**: Perodua, Proton models and vans can be wrongly scored as Civic, because the "not Civic" photos came from web searches of other sedans. Next: add photos of these as negatives and retrain.
- **No "I don't know" answer**: random noise scores 89% "Civic". The detector in front prevents this in practice.
- **Small test set** (43 photos): good for comparing versions, too small for a precise accuracy number.
- Next learning steps: train my own detector on hand-labelled Civic boxes, then semantic segmentation.

## What's not in this repo (and why)
- **The training photos**: they were downloaded from the web and belong to their owners. The download and filtering approach is described above.
- **The model files**: kept private. Happy to share for an interview.

---

**Tools:** Python, Keras 3 (PyTorch backend), PyTorch, EfficientNetV2, ConvNeXt, Ultralytics YOLOv8, ONNX / ONNX Runtime, NumPy, OpenCV, Matplotlib, Jupyter.

**Author:** Tan Zi Hong (Adam). [LinkedIn](https://www.linkedin.com/in/tan-zi-hong)
