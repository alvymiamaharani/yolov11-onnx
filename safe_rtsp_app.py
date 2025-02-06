import sys
import cv2
import numpy as np
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget, QMessageBox
import machineid
import hashlib

from YoloONNX import YoloONNX


def get_machine_id():
    machine_id = machineid.id()
    return hashlib.sha256(machine_id.encode()).hexdigest()


MACHINE_ID = get_machine_id()
ALLOWED_MACHINE_ID = [
    "5f9c9c4f4dc70401787ade67f04ac31e506e87c4ff4a512b961cc3dca15d2029"]


class YOLOv8RTSPApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # Initialize the YOLOv8 object detector
        self.model_path = "./model/best-240.onnx"
        self.yolov8_detector = YoloONNX(
            self.model_path, conf_thres=0.7, iou_thres=0.7)

        # Initialize RTSP stream
        self.rtsp_url = "rtsp://127.0.0.1:8554/live"  # Replace with your RTSP URL
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

        # Timer to update the GUI every 20 ms (for real-time video)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(20)

    def update_frame(self):
        ret, frame = self.cap.read()

        if not ret:
            print("Failed to grab frame from RTSP stream")
            return

        # Detect objects
        boxes, scores, class_ids = self.yolov8_detector(frame)

        # Draw detections on the frame
        combined_img = self.yolov8_detector.draw_detections(frame)

        # Convert BGR to RGB
        combined_img = cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB)

        # Convert frame to QImage for PyQt5
        height, width, channel = combined_img.shape
        bytes_per_line = 3 * width
        q_img = QImage(combined_img.data, width, height,
                       bytes_per_line, QImage.Format_RGB888)

        # Convert QImage to QPixmap and display it on the label
        self.video_label.setPixmap(QPixmap.fromImage(q_img))

    def closeEvent(self, event):
        # Release resources when closing the application
        self.cap.release()
        event.accept()


def main():
    if MACHINE_ID not in ALLOWED_MACHINE_ID:
        # Tampilkan pesan error jika Machine ID tidak diizinkan
        app = QApplication(sys.argv)

        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setText("Your Device is Not Allowed")
        msg.setWindowTitle("Error")
        msg.exec_()

        sys.exit(1)  # Keluar setelah menampilkan error

    app = QApplication(sys.argv)
    window = YOLOv8RTSPApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
