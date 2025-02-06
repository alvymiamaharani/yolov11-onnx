import sys
import cv2
import numpy as np
import time
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget

from YoloONNXQuantized import YoloONNXQuantized


class YOLOv8RTSPApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # Load model quantized
        self.model_path = "./model/quant320.onnx"
        self.yolov8_detector = YoloONNXQuantized(self.model_path, conf_thres=0.7, iou_thres=0.7)

        # Initialize RTSP stream
        self.rtsp_url = "rtsp://127.0.0.1:8554/live"  # Ganti dengan URL RTSP yang benar
        self.cap = cv2.VideoCapture(self.rtsp_url)

        if not self.cap.isOpened():
            print("Error: Tidak dapat membuka RTSP stream!")

        # Set up the GUI
        self.setWindowTitle("YOLOv8 Quantized - RTSP Stream")
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

        # Timer untuk update frame setiap 20 ms
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(20)

    def update_frame(self):
        ret, frame = self.cap.read()

        if not ret:
            print("Gagal mengambil frame dari RTSP stream")
            return

        # Hitung waktu inference
        start_time = time.time()

        # Deteksi objek
        boxes, scores, class_ids = self.yolov8_detector.detect_objects(frame)

        # Gambar bounding box hanya border (tanpa warna)
        output_img = self.yolov8_detector.draw_detections(frame)

        # Konversi BGR ke RGB untuk PyQt5
        output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)

        # Convert frame ke QImage untuk ditampilkan di PyQt5
        height, width, channel = output_img.shape
        bytes_per_line = 3 * width
        q_img = QImage(output_img.data, width, height, bytes_per_line, QImage.Format_RGB888)

        # Tampilkan gambar ke QLabel
        self.video_label.setPixmap(QPixmap.fromImage(q_img))

    def closeEvent(self, event):
        # Hentikan capture ketika aplikasi ditutup
        self.cap.release()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = YOLOv8RTSPApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
