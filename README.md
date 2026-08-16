# Phát hiện đám cháy từ ảnh vệ tinh

## Giới thiệu đề tài

Cháy rừng và các đám cháy ngoài trời có thể gây thiệt hại lớn đến môi trường, hệ sinh thái, tài sản và sức khỏe con người. Việc phát hiện sớm khu vực có cháy đóng vai trò quan trọng trong công tác cảnh báo, giám sát và hỗ trợ lực lượng chức năng đưa ra phương án ứng phó kịp thời.

Ảnh vệ tinh có khả năng bao phủ những khu vực rộng, kể cả các vùng khó tiếp cận bằng phương pháp quan sát trực tiếp. Vì vậy, đề tài này hướng đến việc ứng dụng **Computer Vision** và **Deep Learning** để xây dựng một hệ thống có khả năng nhận diện, định vị khu vực xuất hiện đám cháy trên ảnh vệ tinh.

Đây là dự án nghiên cứu và thực nghiệm ở giai đoạn khởi đầu. Trọng tâm ban đầu là xác định bài toán, chuẩn bị dữ liệu và khảo sát các hướng tiếp cận phù hợp; chưa đặt mục tiêu triển khai thành một hệ thống cảnh báo cháy vận hành trong thực tế.

## Bài toán đặt ra

Đầu vào của hệ thống là ảnh vệ tinh hoặc ảnh viễn thám quang học. Đầu ra mong muốn là vị trí của khu vực nghi ngờ có cháy trên ảnh, được biểu diễn bằng:

- **Bounding box** đối với bài toán phát hiện đối tượng.
- **Mask** đối với bài toán phân đoạn vùng cháy, nếu dữ liệu gán nhãn phù hợp.

Bài toán cần phân biệt vùng cháy với các đối tượng có đặc điểm thị giác tương tự như mây, sương, vùng đất sáng, ánh nắng phản chiếu hoặc khu vực có màu sắc gần giống khói và lửa.

## Mục tiêu của dự án

- Tìm hiểu đặc điểm của ảnh vệ tinh trong bài toán nhận diện đám cháy.
- Xây dựng quy trình thu thập, kiểm tra, tiền xử lý và tăng cường dữ liệu ảnh.
- Huấn luyện mô hình phát hiện vị trí đám cháy trên ảnh.
- Khảo sát và so sánh một số kiến trúc phát hiện đối tượng phổ biến.
- Đánh giá mô hình bằng các chỉ số phù hợp như Precision, Recall, mAP và thời gian suy luận.
- Tạo nền tảng để có thể mở rộng sang phân đoạn vùng cháy hoặc xây dựng hệ thống cảnh báo trong tương lai.

## Câu hỏi nghiên cứu

Dự án được định hướng bởi một số câu hỏi chính:

1. Mô hình học sâu có thể phát hiện đám cháy trên ảnh vệ tinh với mức độ tin cậy như thế nào?
2. Các kiến trúc phát hiện đối tượng một giai đoạn và hai giai đoạn có ưu, nhược điểm gì đối với dữ liệu này?
3. Việc bổ sung ảnh không có đám cháy và áp dụng tăng cường dữ liệu có giúp giảm cảnh báo nhầm hay không?
4. Độ chính xác và tốc độ suy luận cần được cân bằng như thế nào nếu hệ thống được phát triển theo hướng cảnh báo sớm?

## Phạm vi ban đầu

Trong phạm vi đề tài, dự án tập trung vào:

- Ảnh vệ tinh hoặc ảnh viễn thám ở dạng RGB.
- Một lớp đối tượng chính là `fire`.
- Phát hiện đám cháy trên từng ảnh độc lập.
- Thực nghiệm với các kiến trúc như YOLOv8, RT-DETR, Faster R-CNN, RetinaNet và SSD.
- Khảo sát thêm hướng phân đoạn vùng cháy bằng YOLOv8 Segmentation.

Dự án chưa tập trung vào dữ liệu đa phổ, chuỗi ảnh theo thời gian, thông tin địa lý từ GeoTIFF, điều kiện khí tượng hoặc việc xác định tọa độ cháy ngoài thực địa. Đây có thể là các hướng mở rộng sau này.

## Dữ liệu dự kiến

Dữ liệu phục vụ dự án gồm hai nhóm chính:

- Ảnh có xuất hiện đám cháy, được gán nhãn vị trí vùng cháy.
- Ảnh không có đám cháy, được sử dụng để giúp mô hình học cách phân biệt nền và hạn chế cảnh báo nhầm.

Trước khi huấn luyện, dữ liệu dự kiến sẽ trải qua các bước kiểm tra ảnh lỗi, loại bỏ ảnh trùng lặp, chuẩn hóa định dạng, chia tập train/validation/test và tăng cường dữ liệu. Việc chia tập cần được thực hiện trước khi tạo các biến thể tăng cường để hạn chế rò rỉ dữ liệu giữa các tập.

## Hướng tiếp cận đề xuất

Quy trình nghiên cứu dự kiến gồm các giai đoạn:

1. Khảo sát và chuẩn bị dữ liệu ảnh vệ tinh.
2. Gán nhãn khu vực có cháy theo định dạng phù hợp với từng mô hình.
3. Tiền xử lý, kiểm tra chất lượng và tăng cường dữ liệu.
4. Xây dựng mô hình cơ sở để làm mốc so sánh.
5. Huấn luyện và tinh chỉnh các kiến trúc được lựa chọn.
6. Đánh giá định lượng trên cùng một tập kiểm thử.
7. Phân tích lỗi, đặc biệt là các trường hợp bỏ sót đám cháy và cảnh báo nhầm.
8. Đề xuất hướng cải tiến và khả năng triển khai trong tương lai.

## Công nghệ dự kiến sử dụng

- Python
- Jupyter Notebook hoặc Google Colab
- PyTorch
- Ultralytics
- Detectron2
- OpenCV và Albumentations
- Roboflow cho quản lý và chuyển đổi định dạng dữ liệu

Danh sách công nghệ có thể được điều chỉnh trong quá trình thực hiện tùy theo yêu cầu của dữ liệu và tài nguyên tính toán.

## Cài đặt và chạy

```powershell
python -m pip install -r requirements.txt
python -m pip install -e ".[dev,notebooks]"
```

Detectron2 không được khóa trong dependency chung vì wheel phụ thuộc hệ điều hành,
phiên bản PyTorch và CUDA. Hãy cài bản Detectron2 tương thích trước khi chạy hai
notebook Faster R-CNN/RetinaNet.

Các lệnh chính:

```powershell
fire-prepare-data --input data/raw --output data/interim/clean_images
fire-validate-data --images <images> --labels <labels>
fire-split-data --images <images> --labels <labels> --output data/processed/detection --group-manifest <groups.csv>
fire-augment-negatives --input <negative-train-images> --output <augmented-images> --labels-output <augmented-labels>
fire-train --config configs/models/yolov8n.yaml
fire-evaluate --config configs/models/yolov8n.yaml --checkpoint <best.pt>
fire-predict --checkpoint <best.pt> --source <image-or-directory>
```

`group-manifest` có định dạng `image,group`. Các ảnh/tiles thuộc cùng scene phải có cùng `group`. Dataset và artifact lớn không được đưa vào Git; xem thêm `data/README.md` và `artifacts/README.md`.

## Kết quả mong đợi

- Một bộ dữ liệu được tổ chức và kiểm tra rõ ràng cho bài toán phát hiện đám cháy.
- Một quy trình thực nghiệm có thể tái sử dụng cho nhiều kiến trúc mô hình.
- Mô hình có khả năng xác định vị trí đám cháy trên ảnh đầu vào.
- Báo cáo so sánh các mô hình dựa trên độ chính xác, khả năng phát hiện và tốc độ xử lý.
- Phân tích các giới hạn của phương pháp và đề xuất hướng phát triển tiếp theo.

## Hạn chế cần lưu ý

Kết quả của mô hình phụ thuộc nhiều vào chất lượng và mức độ đa dạng của dữ liệu. Ảnh vệ tinh có thể chịu ảnh hưởng bởi mây, khói, độ phân giải, góc chụp, thời điểm chụp và điều kiện ánh sáng. Do đó, đầu ra của dự án chỉ nên được xem là thông tin hỗ trợ nghiên cứu hoặc cảnh báo sơ bộ, không thay thế kết luận của cơ quan chuyên môn.

## Trạng thái

**Giai đoạn xây dựng pipeline thực nghiệm có thể tái lập.**

Source code hiện hỗ trợ chuẩn bị/chia dữ liệu, huấn luyện, đánh giá và suy luận YOLO. Faster R-CNN, RetinaNet và SSDLite vẫn là notebook experiment; cần chạy lại trên cùng phiên bản dataset trước khi đưa ra kết luận so sánh.
