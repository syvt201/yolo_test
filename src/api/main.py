from fastapi import FastAPI, UploadFile
import cv2
import numpy as np
from ..inference import yolo_inference
import base64
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
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
    img_bytes = await file.read()

    img = cv2.imdecode(
        np.frombuffer(img_bytes,np.uint8),
        cv2.IMREAD_COLOR
    )

    yolo_result, viz_640, viz_org = yolo_inference.process(img)

    viz640_b64 = encode_image(viz_640)
    vizorg_b64 = encode_image(viz_org)

    return {
        "result": yolo_result.tolist(),
        "viz_640": viz640_b64,
        "viz_org": vizorg_b64
    }