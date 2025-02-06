import cv2
from YoloONNXQuantized import YoloONNXQuantized

# Path gambar dan model
img_path = "./images/testing.png"
model_path = "./model/quant320.onnx"  # Model quantized

# Inisialisasi model YoloONNXQuantized
yolo_quant = YoloONNXQuantized(model_path, conf_thres=0.7, iou_thres=0.7)

# Baca gambar
image = cv2.imread(img_path)

# Deteksi objek
boxes, scores, class_ids = yolo_quant.detect_objects(image)

# Gambar bounding box hanya border (tanpa warna)
output_img = yolo_quant.draw_detections(image)

# Simpan hasil
cv2.imwrite("./images/testing_result_quantized.png", output_img)

# Tampilkan hasil (opsional, jika Anda ingin melihatnya langsung)
# cv2.imshow("Deteksi Objek Quantized", output_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
