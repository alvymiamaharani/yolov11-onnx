import sys
import cv2
import numpy as np
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QGridLayout, QWidget

from YoloONNX import YoloONNX


class YOLOv8RTSPApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # Initialize the YOLOv8 object detector
        self.model_path = "./model/best-240.onnx"
        self.yolov8_detector = YoloONNX(
            self.model_path, conf_thres=0.7, iou_thres=0.7)

        # Initialize RTSP streams
        self.rtsp_urls = [
            "rtsp://127.0.0.1:8554/live1",
            "rtsp://127.0.0.1:8554/live2",
            "rtsp://127.0.0.1:8554/live3"
        ]
        self.caps = [cv2.VideoCapture(url) for url in self.rtsp_urls]

        # Set up the GUI
        self.setWindowTitle("YOLOv8 Object Detection - RTSP Streams")
        self.setGeometry(100, 100, 1280, 720)

        # Create labels to display the videos
        self.video_labels = [QLabel(self) for _ in self.rtsp_urls]
        for label in self.video_labels:
            label.setAlignment(Qt.AlignCenter)

        # Create a grid layout and set it for the main window
        layout = QGridLayout()
        layout.addWidget(self.video_labels[0], 0, 0)
        layout.addWidget(self.video_labels[1], 0, 1)
        # Full width at bottom
        layout.addWidget(self.video_labels[2], 1, 0, 1, 2)

        container = QWidget(self)
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Timer to update the GUI every 20 ms (for real-time video)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frames)
        self.timer.start(20)

    def update_frames(self):
        for i, cap in enumerate(self.caps):
            ret, frame = cap.read()
            if not ret:
                print(
                    f"Failed to grab frame from RTSP stream {self.rtsp_urls[i]}")
                continue

            # Detect objects
            boxes, scores, class_ids = self.yolov8_detector(frame)
            combined_img = self.yolov8_detector.draw_detections(frame)

            # Convert BGR to RGB
            combined_img = cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB)

            # Resize frame to 16:9 aspect ratio
            target_width = 640
            target_height = int(target_width * 9 / 16)
            combined_img = cv2.resize(
                combined_img, (target_width, target_height))

            # Convert frame to QImage for PyQt5
            height, width, channel = combined_img.shape
            bytes_per_line = 3 * width
            q_img = QImage(combined_img.data, width, height,
                           bytes_per_line, QImage.Format_RGB888)

            # Convert QImage to QPixmap and display it on the label
            self.video_labels[i].setPixmap(QPixmap.fromImage(q_img))

    def closeEvent(self, event):
        # Release resources when closing the application
        for cap in self.caps:
            cap.release()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = YOLOv8RTSPApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
