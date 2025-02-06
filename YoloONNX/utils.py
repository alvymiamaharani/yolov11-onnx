import cv2
import numpy as np

# Daftar nama kelas (sesuaikan dengan model yang digunakan)
class_names = ['Holding stairs', 'Not holding stairs']


def xywh2xyxy(boxes: np.ndarray) -> np.ndarray:
    """
    Convert bounding boxes from center-width-height (cx, cy, w, h) to (x1, y1, x2, y2) format.
    """
    x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return np.stack([x1, y1, x2, y2], axis=1)


def draw_detections(image, boxes, scores, class_ids, mask_alpha=0.3):
    det_img = image.copy()

    img_height, img_width = image.shape[:2]
    font_size = min([img_height, img_width]) * 0.0006
    text_thickness = int(min([img_height, img_width]) * 0.001)

    # Draw bounding boxes and labels of detections
    for class_id, box, score in zip(class_ids, boxes, scores):
        if class_id == 0:  # holding_stairs
            color = (0, 255, 0)  # green
        elif class_id == 1:  # not_holding_stairs
            color = (0, 0, 255)  # red

        draw_box(det_img, box, color)

        label = class_names[class_id]
        # Replacing underscore with space
        caption = f'{label} {int(score * 100)}%'
        draw_text(det_img, caption, box, color, font_size, text_thickness)

    return det_img


def draw_box(image: np.ndarray, box: np.ndarray, color: tuple[int, int, int] = (255, 255, 255),
             thickness: int = 2) -> np.ndarray:
    """
    Draws a bounding box without fill color.
    """
    x1, y1, x2, y2 = box.astype(int)
    return cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)


def draw_text(image: np.ndarray, text: str, box: np.ndarray, color: tuple[int, int, int],
              font_size: float, thickness: int) -> None:
    """
    Draws text above the bounding box.
    """
    x1, y1, _, _ = box.astype(int)
    text_size = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_size, thickness)[0]
    text_x, text_y = x1, y1 - 10

    cv2.putText(image, text, (text_x, max(text_y, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, font_size, color, thickness, cv2.LINE_AA)


def multiclass_nms(boxes, scores, class_ids, iou_threshold):
    """
    Performs Non-Maximum Suppression (NMS) for multi-class object detection.
    """
    if len(boxes) == 0:
        return np.array([], dtype=int)

    unique_classes = np.unique(class_ids)
    keep_indices = []

    for cls in unique_classes:
        cls_indices = np.where(class_ids == cls)[0]
        cls_boxes = boxes[cls_indices]
        cls_scores = scores[cls_indices]

        indices = cv2.dnn.NMSBoxes(
            cls_boxes.tolist(), cls_scores.tolist(), 0.0, iou_threshold)
        if len(indices) > 0:
            keep_indices.extend(cls_indices[indices.flatten()])

    return np.array(keep_indices, dtype=int)
