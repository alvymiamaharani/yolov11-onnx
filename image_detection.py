import cv2
from YoloONNX import YoloONNX

# Path gambar dan model
img_path = "./images/testing.png"
model_path = "./model/best-480.onnx"

# Inisialisasi model YoloONNX
yolo = YoloONNX(model_path, conf_thres=0.7, iou_thres=0.7)

# Baca gambar
image = cv2.imread(img_path)

# Deteksi objek
boxes, scores, class_ids = yolo(image)

# Gambar bounding box hanya border (tanpa warna)
output_img = yolo.draw_detections(image)

# Simpan hasil
cv2.imwrite("./images/testing_result.png", output_img)

# Tampilkan hasil
# cv2.imshow("Deteksi Objek", output_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
