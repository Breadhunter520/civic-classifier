"""
Civic checker web server (FastAPI).

    GET  /          -> the drag-and-drop page (index.html)
    POST /predict   -> photo in, JSON out: every vehicle's box + Civic score (and the whole-photo score)
    GET  /health    -> "ok" (the host checks this to know the server is alive)

Run on the laptop:   MODEL_DIR=../../../hf_space/onnx uvicorn app:app --port 8000
On Render:           models are downloaded from the private Hugging Face page with the HF_TOKEN secret.
Photos are processed in memory and never saved.
"""
import io
import os
import threading
import time

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from PIL import Image, ImageOps

import pipeline as P

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_SIDE = 1600              # big phone photos are shrunk first: YOLO only sees 640 px anyway
MAX_BYTES = 15 * 1024 * 1024
app = FastAPI(title="Civic checker")
_lock = threading.Lock()     # one photo at a time: the free server has very little memory and CPU


@app.get("/")
def page():
    return FileResponse(os.path.join(HERE, "index.html"))


@app.get("/health")
def health():
    return PlainTextResponse("ok")


@app.post("/predict")
async def predict(photo: UploadFile = File(...)):
    data = await photo.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Photo too large (max 15 MB).")
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")   # respect phone rotation
    except Exception:
        raise HTTPException(400, "That file is not a photo I can read.")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    t0 = time.time()
    with _lock:
        cars = P.find_civics(img)
        whole = P.civic_score(img) if cars else None   # no vehicle found -> don't ask the Civic model
    return {
        "width": img.width, "height": img.height,
        "cars": [{"box": [round(v, 1) for v in c["box"]], "yolo": c["yolo"], "yolo_conf": round(c["yolo_conf"], 3),
                  "civic": round(c["civic"], 4)} for c in cars],
        "whole_photo_civic": None if whole is None else round(whole, 4),
        "seconds": round(time.time() - t0, 2),
    }
