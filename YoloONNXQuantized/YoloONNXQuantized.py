import cv2
import numpy as np
import onnxruntime
import time

# Daftar nama kelas (sesuaikan dengan model yang digunakan)
class_names = ['Holding stairs', 'Not holding stairs']

class YoloONNXQuantized:
    def __init__(self, model_path, conf_thres=0.5, iou_thres=0.5):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres

        # Load ONNX model
        self.session = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.get_model_details()

    def detect_objects(self, image):
        """
        Deteksi objek pada gambar + hitung inference time.
        """
        start_time = time.time()  # Mulai waktu inferensi
        input_tensor = self.prepare_input(image)
        outputs = self.inference(input_tensor)
        self.boxes, self.scores, self.class_ids = self.process_output(outputs, image)
        self.inference_time = (time.time() - start_time) * 1000  # Konversi ke ms
        return self.boxes, self.scores, self.class_ids

    def prepare_input(self, image):
        """
        Preprocess input image: Convert to RGB, resize, normalize, and reshape for model input.
        """
        self.img_height, self.img_width = image.shape[:2]
        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))
        input_img = input_img / 255.0  # Normalize to [0,1]
        input_img = input_img.transpose(2, 0, 1)[np.newaxis, :, :, :].astype(np.float32)
        return input_img

    def inference(self, input_tensor):
        """
        Lakukan inference dengan ONNX model.
        """
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})
        return outputs

    def process_output(self, output, image):
        """
        Proses hasil output model untuk mendapatkan bounding box, skor, dan ID kelas.
        """
        predictions = np.squeeze(output[0]).T
        scores = np.max(predictions[:, 4:], axis=1)
        keep = scores > self.conf_threshold

        if not np.any(keep):
            return [], [], []

        predictions, scores = predictions[keep], scores[keep]
        class_ids = np.argmax(predictions[:, 4:], axis=1)
        boxes = self.extract_boxes(predictions, image)

        # Apply Non-Maximum Suppression (NMS)
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), self.conf_threshold, self.iou_threshold)
        indices = indices.flatten() if len(indices) > 0 else []

        return np.array(boxes)[indices], np.array(scores)[indices], np.array(class_ids)[indices]

    def extract_boxes(self, predictions, image):
        """
        Ekstraksi bounding box dan rescale ke dimensi asli gambar.
        """
        img_h, img_w = image.shape[:2]
        x, y, w, h = predictions[:, 0], predictions[:, 1], predictions[:, 2], predictions[:, 3]
        x1 = ((x - w / 2) / self.input_width) * img_w
        y1 = ((y - h / 2) / self.input_height) * img_h
        x2 = ((x + w / 2) / self.input_width) * img_w
        y2 = ((y + h / 2) / self.input_height) * img_h
        return np.stack([x1, y1, x2, y2], axis=1).astype(int)

    def draw_detections(self, image):
        """
        Gambar bounding box, label, dan inference time di gambar.
        """
        for (x1, y1, x2, y2), score, class_id in zip(self.boxes, self.scores, self.class_ids):
            if class_id == 0:  # holding_stairs
                color = (0, 255, 0)  # green
            elif class_id == 1:  # not_holding_stairs
                color = (0, 0, 255)  # red
            else:
                color = (255, 255, 255)
        
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            label = f'{class_names[class_id]}: {score:.2f}'
            cv2.putText(image, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        print(f"Inference time: {self.inference_time:.2f} ms")
        return image

    def get_model_details(self):
        """
        Ambil informasi model ONNX.
        """
        model_inputs = self.session.get_inputs()
        self.input_names = [inp.name for inp in model_inputs]
        self.input_shape = model_inputs[0].shape
        self.input_height, self.input_width = self.input_shape[2:]

        model_outputs = self.session.get_outputs()
        self.output_names = [out.name for out in model_outputs]
