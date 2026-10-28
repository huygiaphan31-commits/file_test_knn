# Tóm tắt tài liệu nguồn — Đề tài số 7

Tóm tắt này dựa trên các file đã kiểm tra trong `source/`. Nội dung ghi **[TÀI LIỆU NGUỒN]** phản ánh những gì tài liệu nêu, không xác nhận rằng các thí nghiệm trong báo cáo mẫu đã được chạy lại.

## Danh sách tài liệu

| File | Phần đã đọc và sử dụng |
|---|---|
| `ĐỀ TÀI SỐ 7.docx` | Đọc nội dung yêu cầu của đề tài; tài liệu không có số trang rõ trong phần trích xuất. |
| `Dethicuoiky_DA_TL_Trituenhantao.pdf` | PDF 22 trang. Đề tài số 7 nằm ở trang in 12–14; yêu cầu sản phẩm và khai báo AI ở trang in 21–22; rubric ở trang in 1–2. |
| `BÁO CÁO ĐỒ ÁN TRÍ TUỆ NHÂN TẠO.docx` | Đọc các phần 1–5, bảng kiểm thử và hình minh họa mã demo. DOCX không có số trang ổn định; dẫn theo tên phần/mục và bảng. |

## Yêu cầu chính thức được tìm thấy

**[TÀI LIỆU NGUỒN — `ĐỀ TÀI SỐ 7.docx`; PDF trang in 12–14]**

- Tìm hiểu khái niệm Computer Vision, vai trò trong AI và một số bài toán phổ biến.
- Chọn một bài toán CV để minh họa hoặc làm demo. Ví dụ gồm nhận diện chữ số viết tay, phân loại chó/mèo, nhận diện biển báo đơn giản, phát hiện khuôn mặt minh họa, phân loại có/không có vật thể và nhận diện trạng thái đeo khẩu trang.
- Mô tả đầu vào, đầu ra, người dùng mục tiêu, phạm vi và dữ liệu/ảnh.
- Trình bày chuẩn bị ảnh, tiền xử lý, resize nếu cần, chuyển ảnh sang dạng phù hợp, xử lý bằng mô hình/thuật toán/công cụ và trả kết quả.
- Demo có thể là chương trình Python, Google Colab, công cụ hoặc mô hình sẵn có, video/hình minh họa hay bảng đầu vào–đầu ra.
- Chuẩn bị ít nhất 05–10 ảnh kiểm thử; cho từng ảnh ghi đầu vào, kết quả, đúng/sai hoặc nhận xét, và lý do xử lý tốt/chưa tốt.
- Nếu dùng mô hình/công cụ sẵn có, nêu tên, đầu vào, cách chạy, cách kiểm chứng và trường hợp dự đoán sai/chưa chính xác.
- Phân tích yếu tố ảnh hưởng, rủi ro, hạn chế, ưu/nhược điểm và hướng phát triển.

**[TÀI LIỆU NGUỒN — PDF trang in 1–2, 21–22]**

- Rubric phân bổ: kết quả đồ án 30%, báo cáo/bố cục 20%, trình bày trực tiếp 30%, làm việc nhóm 20%.
- Phần sản phẩm nộp cuối PDF nêu slide 15–25 trang, báo cáo 20–50 trang; dàn ý có giới thiệu/mục tiêu/phạm vi, mô tả bài toán, cơ sở lý thuyết, mô hình hóa/thiết kế, minh họa/demo, kết quả thực nghiệm, đánh giá, rủi ro dữ liệu/đạo đức/độ tin cậy, kết luận, tài liệu tham khảo và phụ lục.
- Nếu dùng AI, PDF yêu cầu khai báo công cụ, nội dung được AI hỗ trợ, phần nhóm tự chỉnh sửa/kiểm thử và lỗi/hạn chế của kết quả AI nếu có.

## Nội dung báo cáo mẫu

**[TÀI LIỆU NGUỒN — `BÁO CÁO ĐỒ ÁN TRÍ TUỆ NHÂN TẠO.docx`]**

- Phần 1 giới thiệu khái niệm/vai trò CV và liệt kê phân loại ảnh, phát hiện vật thể, nhận diện chữ viết tay/khuôn mặt/biển báo, ảnh y tế và xe tự lái.
- Phần 2 chọn nhận diện chữ số viết tay bằng KNN. Mô tả đầu vào chữ số xám, đầu ra nhãn 0–9 và độ tin cậy, người dùng/phạm vi, dataset Digits và nguyên lý láng giềng gần nhất/Euclidean.
- Phần 3 mô tả chuẩn bị dữ liệu, resize, làm phẳng ảnh 2D thành vector, KNN với `n_neighbors=3` và hiển thị dự đoán.
- Mã trong hình demo gọi `sklearn.datasets.load_digits()`, làm phẳng ảnh, chia train/test 80/20 (`random_state=42`), dùng KNN K=3 và in accuracy. Đây là nội dung mã được trình bày trong hình, không phải kết quả mà Skill đã chạy.
- Phần 4 có bảng **8 mô tả** trường hợp; bảng ghi sáu dự đoán đúng và hai sai (số 8 dự đoán thành 3; số 4 thành 9). Tài liệu không kèm tám ảnh và ground truth có thể kiểm tra độc lập. Không dùng bảng này làm kết quả thực nghiệm của sinh viên nếu chưa tái lập.
- Phần 5 nêu độ phân giải, chất lượng nét và K ảnh hưởng đến kết quả; nêu KNN đơn giản/dễ giải thích nhưng suy luận chậm hơn khi dữ liệu lớn, nhạy với nhiễu/chuẩn hóa và không học đặc trưng phức tạp như CNN; đề xuất HOG/PCA hoặc CNN.

### Chi tiết kỹ thuật cần diễn đạt thận trọng

- Báo cáo nêu kích thước 8×8 hoặc 28×28 và ví dụ vector 8×8=64. Các đối tượng toán học xác định được trong DOCX bao gồm các kích thước này, miền nhãn 0–9, K=3/K=5, 80/20 và `random_state=42`.
- Hình mã dùng `load_digits()` nhưng báo cáo cũng nhắc Digits là phiên bản thu nhỏ 8×8 và đưa cả kích thước 28×28 vào mô tả. Khi thực hiện, phải xác nhận chính xác dữ liệu đầu vào và phép tiền xử lý; không mặc định mã trong hình xử lý trực tiếp ảnh 28×28.
- Mô tả “độ tin cậy” ở phần đầu ra không được minh họa bởi mã demo trong hình; mã chỉ in accuracy trên tập test. Nếu báo cáo đầu ra độ tin cậy, cần triển khai và kiểm chứng riêng.
- Bảng tám mẫu là nội dung có trong báo cáo nguồn, nhưng không có ảnh tương ứng trong tài liệu được đọc. Vì vậy không thể xác nhận nguyên nhân dự đoán hoặc độ chính xác thực tế từ bảng này.

## Giới hạn đọc và cách sử dụng

- Cả hai DOCX đều trích được văn bản; báo cáo có một bảng và một hình chứa mã demo. Kích thước/công thức quan trọng trong báo cáo được lưu dưới dạng đối tượng toán học và đã được đọc riêng.
- PDF có 22 trang và trích xuất được văn bản; đề tài CV nằm ở trang 12–14. Hình trích được ở trang 1 là logo, không phải dữ liệu CV.
- Không phát hiện thiếu trang PDF khi trích xuất văn bản, nhưng trích xuất văn bản không bảo đảm mọi chi tiết định dạng/hình ảnh của bản in đã được kiểm chứng. Nếu cần xác nhận bố cục hoặc hình ảnh cụ thể, mở bản gốc trực quan.
- Tài liệu nguồn không đưa kết quả chạy thực tế có thể tái lập, nguồn/giấy phép chi tiết của dữ liệu, hoặc bộ ảnh kiểm thử kèm nhãn. Sinh viên phải tự bổ sung các phần này bằng bằng chứng thực tế.
- Mọi khuyến nghị kỹ thuật không nêu ở trên phải được ghi là **[KIẾN THỨC BỔ SUNG]**; không bịa số trang cho DOCX.
