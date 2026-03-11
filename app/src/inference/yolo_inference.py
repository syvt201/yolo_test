import onnxruntime as ort
import numpy as np
import cv2
from app.src.api.config import YOLO_ONNX, YOLO_CLASSES
from app.src.inference import utils
import logging

logger = logging.getLogger(__name__)
# session = ort.InferenceSession(YOLO_ONNX)

providers = [
    "CUDAExecutionProvider",
    "CPUExecutionProvider"
]

session = ort.InferenceSession(
    YOLO_ONNX,
    providers=providers
)
input_name = session.get_inputs()[0].name

print("Using provider:", session.get_providers())

logger.info("ONNX session created")

cls_dict = {}
with open(YOLO_CLASSES, "r") as f:
    for line in f:
        cls_id, cls = line.split(",")
        cls_dict[int(cls_id)] = cls.strip()

logger.info("Load %d YOLO's classes", len(cls_dict))

def detect(img):
    img = img.transpose(2,0,1)
    img = img.astype(np.float32) / 255.0
    img = img[None]
    outputs = session.run(None, {input_name: img})

    return img, outputs

def process(img):
    """
    Returns:
        output: YOLO output, shape (no_detections, (x1, y2, x2, y2, score, class_id))
        , viz_resized, viz_org
    """
    logger.info(f"Start processing image, shape={img.shape}")
    if isinstance(img, str):
        logger.info(f"Reading image from {img}")
        img = cv2.imread(img)
    
    logger.info("Letterbox resize: original_shape=(%d,%d), target=(640,640)", img.shape[0], img.shape[1])
    resized_img, gain, (dw,dh) = utils.letterbox(img, (640,640))
    
    _, outputs = detect(resized_img)
    logger.info("Raw detections: %d", len(outputs[0][0]))
    
    output = outputs[0][0]
    logger.info("Apply NMS")
    nms_pred = nms(output)
    logger.info("Detections after NMS: %d", len(nms_pred))
    
    for p in nms_pred:
        utils.draw_bbox(resized_img, p, cls_dict)
        
    boxes = nms_pred[:, :4]
    
    logger.info("Postprocess bboxes")
    boxes_scaled = utils.scale_boxes(
        (640,640),
        boxes.copy(),
        img.shape[:2],
        gain,
        dw, 
        dh
    )
    
    nms_pred[:, :4] = boxes_scaled
    
    for p in nms_pred:
        utils.draw_bbox(img, p, cls_dict,thickness=1, fontScale=1.0)
        
    # viz_resized = resized_img
    # viz_org = cv2.resize(resized_img, (org_width, org_height))
    
    logger.info("Processing finished, final detections=%d", len(nms_pred))
    return nms_pred, resized_img, img

def nms(pred):
    boxes = pred[:,:4].tolist()
    scores = pred[:,4].tolist()

    indices = cv2.dnn.NMSBoxes(
        boxes,
        scores,
        score_threshold=0.2,
        nms_threshold=0.7
    )
    
    final = pred[indices.flatten()]
    
    return final