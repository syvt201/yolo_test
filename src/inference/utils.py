import numpy as np

import cv2
import numpy as np

def letterbox(
    img,
    new_shape=(640, 640),
    color=(114,114,114),
    auto=False,
    scale_fill=False,
    scaleup=True,
    stride=32
):
    shape = img.shape[:2]  # (h,w)

    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # scale ratio
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    if not scaleup:
        r = min(r, 1.0)

    # new size without padding
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))

    # padding
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]

    if auto:
        dw = dw % stride
        dh = dh % stride

    elif scale_fill:
        dw, dh = 0.0, 0.0
        new_unpad = new_shape
        r = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2

    # resize
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    # padding
    top, bottom = int(round(dh-0.1)), int(round(dh+0.1))
    left, right = int(round(dw-0.1)), int(round(dw+0.1))

    img = cv2.copyMakeBorder(
        img,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=color
    )

    return img, r, (dw, dh)

def scale_boxes(img1_shape, boxes, img0_shape):
    """
    Rescale bounding boxes from img1_shape to img0_shape

    img1_shape : shape used for inference (640,640)
    boxes      : numpy array (N,4) -> x1,y1,x2,y2
    img0_shape : original image shape (H,W)
    """

    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])

    pad_w = (img1_shape[1] - img0_shape[1] * gain) / 2
    pad_h = (img1_shape[0] - img0_shape[0] * gain) / 2

    boxes[:, [0,2]] -= pad_w
    boxes[:, [1,3]] -= pad_h

    boxes[:, :4] /= gain

    boxes[:, 0] = np.clip(boxes[:, 0], 0, img0_shape[1])
    boxes[:, 1] = np.clip(boxes[:, 1], 0, img0_shape[0])
    boxes[:, 2] = np.clip(boxes[:, 2], 0, img0_shape[1])
    boxes[:, 3] = np.clip(boxes[:, 3], 0, img0_shape[0])

    return boxes


def draw_bbox(img, pred, cls_dict, thickness=1, fontScale=0.5):
    x1, y1, x2, y2, score, class_id = pred
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), thickness)
    cv2.putText(img, f"{cls_dict[int(class_id)]} {score:.2f}", (int(x2), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 255, 0), thickness=thickness)
