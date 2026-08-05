# Artifacts

Thư mục này chứa đầu ra sinh ra từ thí nghiệm và không phải source code:

- `checkpoints/detection/`, `checkpoints/segmentation/`: `best.pt`, `last.pt`.
- `predictions/`: ảnh và JSON dự đoán.
- `metrics/`: CSV, JSON và đầu ra đánh giá.
- `plots/`: learning curves, PR curve và confusion matrix.
- `exports/`: ONNX, TensorRT, TorchScript hoặc định dạng triển khai khác.

Các tệp lớn trong thư mục này được loại khỏi Git.
