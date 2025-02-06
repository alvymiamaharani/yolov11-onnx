
import sys
import time
import cv2
import numpy as np
import threading
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget

from YoloONNX import YoloONNX


class YOLOv8RTSPApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # Initialize the YOLOv8 object detector
        self.model_path = "./model/best-240.onnx"
        self.yolov8_detector = YoloONNX(
            self.model_path, conf_thres=0.7, iou_thres=0.7)

        # Initialize RTSP stream
        self.rtsp_url = "rtsp://127.0.0.1:8554/live"  # Ganti dengan RTSP URL Anda
        self.cap = cv2.VideoCapture(self.rtsp_url)

        # Set up the GUI
        self.setWindowTitle("YOLOv8 Object Detection - RTSP Stream")
        self.setGeometry(100, 100, 800, 600)

        # Create a label to display the video
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)

        # Create a layout and set it for the main window
        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        container = QWidget(self)
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Timer to update GUI every 20ms (for real-time video)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(20)  # 50 FPS

        # Variabel untuk menyimpan hasil inference
        self.last_frame = None
        self.last_detections = None
        self.lock = threading.Lock()  # Lock untuk sinkronisasi data antara thread

        # Thread untuk menjalankan inference
        self.inference_thread = threading.Thread(
            target=self.run_inference, daemon=True)
        self.inference_thread.start()

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to grab frame from RTSP stream")
            return

        # Simpan frame terbaru untuk inference di thread terpisah
        with self.lock:
            self.last_frame = frame.copy()

        # Gunakan hasil inference terbaru jika sudah ada
        with self.lock:
            if self.last_detections is not None:
                boxes, scores, class_ids = self.last_detections
                frame = self.yolov8_detector.draw_detections(frame)

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert frame to QImage for PyQt5
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(frame.data, width, height,
                       bytes_per_line, QImage.Format_RGB888)

        # Convert QImage to QPixmap and display it on the label
        self.video_label.setPixmap(QPixmap.fromImage(q_img))

    def run_inference(self):
        while True:
            # Batasi kecepatan inference agar tidak membebani CPU
            time.sleep(0.02)
            with self.lock:
                if self.last_frame is None:
                    continue
                frame = self.last_frame.copy()

            # Jalankan inference
            boxes, scores, class_ids = self.yolov8_detector(frame)

            # Simpan hasil inference
            with self.lock:
                self.last_detections = (boxes, scores, class_ids)

    def closeEvent(self, event):
        # Release resources when closing the application
        self.cap.release()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = YOLOv8RTSPApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
