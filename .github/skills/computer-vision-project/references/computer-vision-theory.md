# Kiến thức nền Computer Vision

> Nội dung trong file này là **[KIẾN THỨC BỔ SUNG]** để giải thích nền tảng cho sinh viên. Các ví dụ bài toán khớp với chủ đề trong đề bài nhưng phần giải thích dưới đây không phải trích nguyên văn từ tài liệu nguồn. Đối chiếu yêu cầu chính thức trong [tóm tắt nguồn](./source-summary.md).

## 1. Computer Vision là gì?

Computer Vision (thị giác máy tính) là lĩnh vực xây dựng phương pháp để máy tính lấy thông tin từ ảnh hoặc video và dùng thông tin đó cho một nhiệm vụ. Ảnh số thường được biểu diễn bằng pixel; mỗi pixel có một hoặc nhiều giá trị màu/sáng. Mô hình không “nhìn” như con người: nó xử lý biểu diễn số và học quy luật từ dữ liệu hoặc dùng quy tắc được thiết kế.

## 2. Các dạng bài toán thường gặp

| Bài toán | Đầu ra điển hình | Ví dụ |
|---|---|---|
| Phân loại ảnh (classification) | Một hoặc nhiều nhãn cho cả ảnh | Chữ số là 3; ảnh là chó hay mèo |
| Phát hiện vật thể (object detection) | Nhãn và hộp giới hạn cho từng vật thể | Vị trí biển báo trong ảnh |
| Phân đoạn ảnh (segmentation) | Nhãn/mặt nạ theo pixel | Vùng đường đi hoặc vùng tổn thương |
| Nhận diện chữ (OCR/handwriting recognition) | Chuỗi ký tự hoặc chữ số | Đọc chữ viết tay |
| Nhận diện danh tính khuôn mặt | So khớp danh tính hoặc xác minh | Cần đặc biệt cân nhắc quyền riêng tư |

Đề bài nguồn nêu phân loại, detection, chữ viết tay, khuôn mặt, biển báo, ảnh y tế và xe tự lái. Phân đoạn và OCR là phần bổ sung để phân biệt các loại đầu ra.

## 3. Quy trình ảnh cơ bản

1. **Xác định nhiệm vụ và nhãn:** mô tả rõ điều gì được đưa vào và hệ thống cần trả về gì.
2. **Thu thập/chọn dữ liệu:** ghi nguồn, quyền sử dụng, điều kiện chụp và cách gán nhãn. Kiểm tra dữ liệu có đại diện cho tình huống dự kiến không.
3. **Khám phá dữ liệu:** xem số lượng ảnh, kích thước, định dạng, nhãn, lớp ít/nhiều và ảnh lỗi.
4. **Tiền xử lý:** các thao tác như đổi kích thước, đổi màu, chuẩn hóa cường độ, cắt vùng quan tâm hoặc làm sạch có thể cần thiết. Chỉ dùng thao tác phù hợp với mô hình và tài liệu mô hình; áp dụng nhất quán cho train và inference.
5. **Chia tập:** tách train/validation/test trước khi điều chỉnh mô hình; tránh để bản sao hoặc ảnh gần trùng rơi vào các tập khác nhau.
6. **Huấn luyện hoặc nạp mô hình:** ghi tên/phiên bản, tham số, nguồn trọng số và phần dữ liệu sử dụng.
7. **Suy luận:** đưa ảnh chưa dùng huấn luyện qua đúng pipeline; lưu dự đoán cùng ID ảnh và cấu hình.
8. **Đánh giá và phân tích lỗi:** so với nhãn đúng đã xác nhận; xem cả các trường hợp thành công và thất bại.
9. **Trình bày:** cho thấy ảnh đầu vào, kết quả mô hình, nhãn đúng (nếu có) và ghi chú về giới hạn.

## 4. Mô hình và lựa chọn

- **KNN:** lưu các ví dụ huấn luyện; khi có ảnh mới, so sánh biểu diễn của ảnh với các mẫu đã lưu và bỏ phiếu từ láng giềng gần. Dễ minh họa, nhưng kết quả phụ thuộc biểu diễn, chuẩn hóa, khoảng cách và K; dự đoán có thể chậm khi mẫu nhiều.
- **CNN:** mạng nơ-ron tích chập thường dùng cho ảnh để học đặc trưng không gian; cần dữ liệu/tài nguyên phù hợp. Có thể dùng mô hình huấn luyện sẵn hoặc fine-tune, nhưng phải nêu rõ nguồn mô hình và pipeline.
- **Transfer learning:** dùng trọng số đã học làm điểm bắt đầu cho bài toán mới. Tên mô hình, nhãn, preprocessing và giấy phép cần được xác minh từ tài liệu của mô hình thực tế.
- **HOG/PCA:** các ví dụ trích xuất/giảm chiều được báo cáo nguồn nêu như hướng phát triển cho KNN; chúng không tự thay thế bước đánh giá.

Không chọn mô hình theo độ “hiện đại” một cách máy móc. Cân bằng tính giải thích, kích thước dữ liệu, tốc độ, yêu cầu cài đặt và khả năng tái lập.

## 5. Đánh giá

- **Accuracy:** tỷ lệ dự đoán đúng trên tập có nhãn. Có thể gây hiểu nhầm khi các lớp mất cân bằng.
- **Precision/Recall/F1:** hữu ích cho phân loại khi cần xem lỗi dương tính giả/âm tính giả hoặc lớp hiếm.
- **Confusion matrix:** cho biết cặp lớp hay nhầm.
- Detection/segmentation cần chỉ số riêng (ví dụ IoU/mAP) và cách định nghĩa nhãn; chỉ dùng khi hiểu và có ground truth phù hợp.

Các chỉ số trên là **[KIẾN THỨC BỔ SUNG]**, không phải danh sách bắt buộc cụ thể trong yêu cầu Đề tài số 7. Tập 5–10 ảnh dùng để minh họa từng trường hợp là quá nhỏ để khẳng định hiệu năng tổng quát. Báo cáo riêng số liệu từ test set chuẩn và kết quả trên các ảnh demo.

## 6. Yếu tố ảnh hưởng và rủi ro

Độ phân giải, ánh sáng, góc chụp, nền, che khuất, nhiễu, nét ảnh, cách gán nhãn, mất cân bằng lớp, rò rỉ dữ liệu và sai khác giữa dữ liệu huấn luyện với ảnh thực tế đều có thể ảnh hưởng dự đoán. Với khuôn mặt/ảnh y tế, cân nhắc sự đồng ý, dữ liệu cá nhân, thiên lệch, giới hạn sử dụng và không đưa ra kết luận chuyên môn vượt quá phạm vi demo.
