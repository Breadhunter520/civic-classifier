"""
Civic finder without PyTorch / Keras / Ultralytics — only numpy, OpenCV, Pillow and ONNX Runtime.
Same steps as civic_detect.py, but every step Ultralytics used to hide is written out here.
Tested against the original Ultralytics + Keras version: identical boxes and scores on 44 photos.

    from pipeline import find_civics, whole_photo
    cars = find_civics(Image.open("street.jpg"))

  1. letterbox   shrink the long side to 640 (keep the shape), grey padding only up to a multiple of 32
  2. YOLO        yolov8n.onnx -> one raw guess per grid spot (6,300 for a 640 x 480 photo),
                 each = 4 box numbers + 80 class scores
  3. cleanup     keep car / bus / truck >= 40%, then NMS (keep the surest box, delete boxes
                 overlapping it by more than 70%, repeat)
  4. crop        each car box + 10% margin, squashed to 384 x 384 like the training photos
  5. Civic       civic.onnx -> one score per car, > 0.5 = 10th-gen Civic
"""
import os
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
VEHICLES = {2: "car", 5: "bus", 7: "truck"}   # COCO class numbers -> names
MIN_CONF, NMS_IOU, MIN_SIDE = 0.4, 0.7, 40      # same settings as civic_detect.py / Ultralytics defaults
CIVIC_SIZE, YOLO_SIZE = 384, 640

MODEL_REPO = "Breadhunter/civic-fc-classifier"   # private Hugging Face model page


def _model_file(name):
    """Use a local copy if MODEL_DIR is set (testing on the laptop); otherwise download it from the
    private model page with the HF_TOKEN secret (on the server). The token is never in the code."""
    local = os.environ.get("MODEL_DIR")
    if local:
        return os.path.join(local, name)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(MODEL_REPO, name, token=os.environ.get("HF_TOKEN"))


_opts = ort.SessionOptions()
_opts.intra_op_num_threads = int(os.environ.get("ORT_THREADS", "1"))   # the free server has a fraction of one CPU
_yolo = ort.InferenceSession(_model_file("yolov8n.onnx"), _opts, providers=["CPUExecutionProvider"])
_civic = ort.InferenceSession(_model_file("civic.onnx"), _opts, providers=["CPUExecutionProvider"])


def resize_bilinear(a, out_h, out_w):
    """Plain bilinear resize (half-pixel centres, no anti-aliasing) = what Keras/PyTorch and OpenCV do.
    Pillow's resize smooths extra when shrinking, which would change the scores slightly."""
    h, w = a.shape[:2]
    ys = np.clip((np.arange(out_h) + 0.5) * h / out_h - 0.5, 0, h - 1)
    xs = np.clip((np.arange(out_w) + 0.5) * w / out_w - 0.5, 0, w - 1)
    y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
    y1, x1 = np.minimum(y0 + 1, h - 1), np.minimum(x0 + 1, w - 1)
    wy, wx = (ys - y0)[:, None, None], (xs - x0)[None, :, None]
    top = a[y0][:, x0] * (1 - wx) + a[y0][:, x1] * wx
    bot = a[y1][:, x0] * (1 - wx) + a[y1][:, x1] * wx
    return (top * (1 - wy) + bot * wy).astype(np.float32)


def civic_score(img):
    """Squash any photo/crop to 384 x 384 (like training) and return the Civic score 0..1."""
    x = resize_bilinear(np.asarray(img.convert("RGB"), dtype=np.float32), CIVIC_SIZE, CIVIC_SIZE)[None]
    return float(_civic.run(None, {_civic.get_inputs()[0].name: x})[0][0][0])


def letterbox(a):
    """Copy of Ultralytics' LetterBox(auto=True): shrink the long side to 640 keeping the shape, then pad
    with grey (114) only up to the next multiple of 32 (e.g. 640 x 480, not a full 640 x 640 square)."""
    h, w = a.shape[:2]
    r = min(YOLO_SIZE / h, YOLO_SIZE / w)
    nh, nw = round(h * r), round(w * r)
    dh, dw = ((YOLO_SIZE - nh) % 32) / 2, ((YOLO_SIZE - nw) % 32) / 2
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    small = a if (nh, nw) == (h, w) else cv2.resize(a.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_LINEAR)  # exactly like Ultralytics
    out = np.full((nh + top + bottom, nw + left + right, 3), 114, dtype=np.float32)
    out[top:top + nh, left:left + nw] = small
    return out, r, left, top


def iou(box, boxes):
    """Overlap between one box and many boxes: shared area / combined area (0 = apart, 1 = identical)."""
    ix = np.clip(np.minimum(box[2], boxes[:, 2]) - np.maximum(box[0], boxes[:, 0]), 0, None)
    iy = np.clip(np.minimum(box[3], boxes[:, 3]) - np.maximum(box[1], boxes[:, 1]), 0, None)
    inter = ix * iy
    area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / (area(box) + area(boxes) - inter)


def nms(boxes, scores):
    """Keep the surest box, delete every box overlapping it by more than NMS_IOU, repeat."""
    order, keep = np.argsort(-scores), []
    while len(order):
        best, rest = order[0], order[1:]
        keep.append(best)
        order = rest[iou(boxes[best], boxes[rest]) <= NMS_IOU]
    return keep


def detect_vehicles(img):
    """YOLO + cleanup -> list of (box in original-photo pixels, class name, confidence)."""
    a = np.asarray(img.convert("RGB"), dtype=np.float32)
    lb, r, left, top = letterbox(a)
    x = (lb / 255.0).transpose(2, 0, 1)[None]                       # 1 x 3 x 640 x 640, values 0..1
    raw = _yolo.run(None, {_yolo.get_inputs()[0].name: x})[0][0]   # 84 x 8400
    cx, cy, bw, bh = raw[:4]
    cls_scores = raw[4:]                                             # 80 x 8400
    cls = cls_scores.argmax(0)
    conf = cls_scores.max(0)
    ok = np.isin(cls, list(VEHICLES)) & (conf >= MIN_CONF)           # cleanup step 1: drop unsure / non-vehicles
    boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)[ok]
    conf, cls = conf[ok], cls[ok]
    keep = []
    for c in np.unique(cls):                                         # cleanup step 2: NMS, per class like Ultralytics
        idx = np.where(cls == c)[0]
        keep += [idx[k] for k in nms(boxes[idx], conf[idx])]
    keep = sorted(keep, key=lambda k: -conf[k])
    H, W = a.shape[:2]
    out = []
    for k in keep:                                                   # undo the letterbox: back to photo pixels
        x0, y0, x1, y1 = boxes[k]
        x0, x1 = np.clip([(x0 - left) / r, (x1 - left) / r], 0, W)
        y0, y1 = np.clip([(y0 - top) / r, (y1 - top) / r], 0, H)
        out.append(((float(x0), float(y0), float(x1), float(y1)), VEHICLES[int(cls[k])], float(conf[k])))
    return out


def find_civics(img):
    """Every vehicle YOLO finds (big enough) -> crop with 10% margin -> Civic score."""
    img = img.convert("RGB")
    W, H = img.size
    cars = []
    for (x0, y0, x1, y1), name, conf in detect_vehicles(img):
        if min(x1 - x0, y1 - y0) < MIN_SIDE:
            continue
        mx, my = 0.10 * (x1 - x0), 0.10 * (y1 - y0)
        crop = img.crop((max(0, x0 - mx), max(0, y0 - my), min(W, x1 + mx), min(H, y1 + my)))
        cars.append({"box": (x0, y0, x1, y1), "yolo": name, "yolo_conf": conf, "civic": civic_score(crop)})
    return cars


def whole_photo(img):
    """Whole-photo mode: YOLO is the 'is there a car?' check, then the Civic model sees the whole photo."""
    has_car = any(min(b[2] - b[0], b[3] - b[1]) >= MIN_SIDE for b, _, _ in detect_vehicles(img))
    return {"car_found": has_car, "civic": civic_score(img)}
