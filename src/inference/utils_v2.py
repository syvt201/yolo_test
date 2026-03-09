import numpy as np
import cv2

def letterbox(
    img,
    new_shape=(640, 640),
    color=(114,114,114),
    scaleup=True,
):
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
        
    scale = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    
    if not scaleup:
        scale = min(scale, 1.0)
    
    new_unpad = (int(round(scale * shape[0])), int(round(scale * shape[1])))
    
    pad_w = new_shape[1] - new_unpad[1]
    pad_h = new_shape[0] - new_unpad[0]
    
    pad_w /= 2
    pad_h /= 2
    
    if shape != new_unpad:
        img = cv2.resize(img, new_unpad[::-1], interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
    
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, color)
    
    return img, (scale, scale), (pad_w, pad_h)

def scale_boxes(img1_shape, boxes, img0_shape):
    """
    Rescale bounding boxes from img1_shape to img0_shape

    img1_shape : shape used for inference (640,640)
    boxes      : numpy array (N,4) -> x1,y1,x2,y2
    img0_shape : original image shape (H,W)
    """
    boxes = boxes.copy()
    
    scale_x = img1_shape[1] / img0_shape[1]
    scale_y = img1_shape[0] / img0_shape[0]
    gain = min(scale_x, scale_y)
    
    new_unpad = (int(round(gain * img0_shape[0])), int(round(gain * img0_shape[1])))
    pad_w = img1_shape[1] - new_unpad[1]
    pad_h = img1_shape[0] - new_unpad[0]
    
    pad_w /= 2
    pad_h /= 2
    
    boxes[:, [0, 2]] -= pad_w
    boxes[:, [1, 3]] -= pad_h
    
    boxes[:, :4] /= gain
    
    boxes[:, [0,2]] = np.clip(boxes[:, [0,2]], 0, img0_shape[1])
    boxes[:, [1,3]] = np.clip(boxes[:, [1,3]], 0, img0_shape[0])
    
    return boxes
    
    