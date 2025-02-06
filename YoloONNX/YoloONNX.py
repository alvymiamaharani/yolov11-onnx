import time
import cv2
import numpy as np
import onnxruntime
from YoloONNX.utils import xywh2xyxy, draw_detections, multiclass_nms


class YoloONNX:
    def __init__(self, model_path, conf_thres=0.7, iou_thres=0.5):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres

        # Load ONNX model
        self.session = onnxruntime.InferenceSession(
            model_path, providers=onnxruntime.get_available_providers())
        self.get_model_details()

    def __call__(self, image):
        return self.detect_objects(image)

    def detect_objects(self, image):
        input_tensor = self.prepare_input(image)
        outputs = self.inference(input_tensor)
        self.boxes, self.scores, self.class_ids = self.process_output(outputs)
        return self.boxes, self.scores, self.class_ids

    def prepare_input(self, image):
        """
        Preprocess input image: Convert to RGB, resize, normalize, and reshape for model input.
        """
        self.img_height, self.img_width = image.shape[:2]
        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_img = cv2.resize(
            input_img, (self.input_width, self.input_height))
        input_img = input_img / 255.0  # Normalize to [0,1]
        input_img = input_img.transpose(
            2, 0, 1)[np.newaxis, :, :, :].astype(np.float32)
        return input_img

    def inference(self, input_tensor):
        """
        Perform inference with the ONNX model.
        """
        start_time = time.perf_counter()
        outputs = self.session.run(
            self.output_names, {self.input_names[0]: input_tensor})
        elapsed_time = (time.perf_counter() - start_time) * 1000
        print(f"Inference time: {elapsed_time:.2f} ms")
        return outputs

    def process_output(self, output):
        """
        Process model output to extract bounding boxes, scores, and class IDs.
        """
        predictions = np.squeeze(output[0]).T
        scores = np.max(predictions[:, 4:], axis=1)
        keep = scores > self.conf_threshold

        if not np.any(keep):
            return [], [], []

        predictions, scores = predictions[keep], scores[keep]
        class_ids = np.argmax(predictions[:, 4:], axis=1)
        boxes = self.extract_boxes(predictions)

        # Apply Non-Maximum Suppression (NMS)
        indices = multiclass_nms(boxes, scores, class_ids, self.iou_threshold)
        return boxes[indices], scores[indices], class_ids[indices]

    def extract_boxes(self, predictions):
        """
        Extract and rescale bounding boxes from model predictions.
        """
        boxes = predictions[:, :4]
        boxes = self.rescale_boxes(boxes)
        return xywh2xyxy(boxes)

    def rescale_boxes(self, boxes):
        """
        Rescale bounding boxes to the original image dimensions.
        """
        scale = np.array([self.input_width, self.input_height,
                         self.input_width, self.input_height])
        boxes = (boxes / scale) * np.array([self.img_width,
                                            self.img_height, self.img_width, self.img_height])
        return boxes

    def draw_detections(self, image):
        """
        Draw bounding boxes on the image (only border, no fill color).
        """
        return draw_detections(image, self.boxes, self.scores, self.class_ids)

    def get_model_details(self):
        """
        Get model input/output details.
        """
        model_inputs = self.session.get_inputs()
        self.input_names = [inp.name for inp in model_inputs]
        self.input_shape = model_inputs[0].shape
        self.input_height, self.input_width = self.input_shape[2:]

        model_outputs = self.session.get_outputs()
        self.output_names = [out.name for out in model_outputs]
