from fastapi import FastAPI, UploadFile
import cv2
import numpy as np
from app.src.inference import yolo_inference
import base64
from fastapi.middleware.cors import CORSMiddleware
from app.src.logging.logging_config import setup_logging
import logging
import time

setup_logging()
logger = logging.getLogger("app.src.api.main")
app = FastAPI()
logger.info("Application started")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # dev only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
def encode_image(img):
    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode("utf-8")

@app.post("/detect")
async def detect(file: UploadFile):
    start_time = time.time()
    logger.info("Received request: %s", file.filename)

    img_bytes = await file.read()

    img = cv2.imdecode(
        np.frombuffer(img_bytes,np.uint8),
        cv2.IMREAD_COLOR
    )
    if img is None:
        logger.error("Image decode failed")
        return {"error": "Invalid image"}
    
    logger.info("Encoded image, (width, height)=(%d, %d)", img.shape[1], img.shape[0])

    try:

        yolo_result, viz_640, viz_org = yolo_inference.process(img)

        viz640_b64 = encode_image(viz_640)
        vizorg_b64 = encode_image(viz_org)

        duration = time.time() - start_time
        logger.info("Request processed in %.3f seconds", duration)

        return {
            "result": yolo_result.tolist(),
            "viz_640": viz640_b64,
            "viz_org": vizorg_b64
        }

    except Exception as e:
        logger.exception("Error during inference: %s", str(e))
        return {"error": "Inference failed"}