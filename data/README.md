# Dữ liệu

- `raw/`: ảnh gốc, chỉ đọc và không chỉnh sửa trực tiếp.
- `external/`: dữ liệu tải từ Roboflow hoặc nguồn ngoài.
- `interim/`: dữ liệu đã lọc nhưng chưa phát hành.
- `processed/detection/`: dữ liệu YOLO detection theo `train/valid/test`.
- `processed/segmentation/`: dữ liệu YOLO segmentation.
- `metadata/`: manifest nguồn, thống kê lớp và danh sách chia tập.

Phải chia ảnh theo ảnh/scene gốc trước khi augmentation. Chỉ tăng cường dữ liệu
trên tập train để tránh rò rỉ dữ liệu.
