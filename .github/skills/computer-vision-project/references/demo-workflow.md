# Quy trình làm demo Computer Vision

Các bước dưới đây là hướng dẫn thực hành **[KIẾN THỨC BỔ SUNG]** để đáp ứng các yêu cầu demo trong [tóm tắt tài liệu nguồn](./source-summary.md). Thay các placeholder bằng lựa chọn đã xác nhận; không coi ví dụ là kết quả thực nghiệm.

## Bước 1 — Chọn một bài toán vừa sức

Viết một câu theo mẫu:

> Với `[đầu vào]`, hệ thống dự đoán/xác định `[đầu ra]` để hỗ trợ `[người dùng]`, trong phạm vi `[giới hạn]`.

Chọn tác vụ có nhãn dễ xác định, ảnh hợp pháp để dùng, đầu ra nhìn thấy được và có đường chạy trên máy cá nhân hoặc Colab. Ví dụ nhận diện chữ số KNN có trong báo cáo mẫu; các lựa chọn khác có trong đề bài. Nêu rõ vì sao chọn và việc gì không làm.

## Bước 2 — Chốt dữ liệu

- Ghi tên dataset hoặc mô tả cách tự thu thập, URL/tài liệu nguồn **chỉ sau khi xác minh**, ngày truy cập nếu cần, giấy phép/quyền sử dụng và số mẫu thực tế.
- Ghi định dạng, kích thước, kênh màu, nhãn và số lượng từng lớp sau khi kiểm tra dữ liệu.
- Nếu tự chụp ảnh, lấy đủ biến thiên có liên quan (góc/ánh sáng/nền), không dùng ảnh người khác khi chưa có quyền.
- Chọn ảnh kiểm thử riêng; không dùng ảnh đã tham gia huấn luyện hay tinh chỉnh để tuyên bố kết quả ngoài mẫu.

## Bước 3 — Chuẩn bị môi trường và kiểm tra đầu vào

- Ghi phiên bản Python, package, mô hình/dataset, OS hoặc Colab, CPU/GPU và lệnh chạy.
- Kiểm tra tệp ảnh tồn tại, định dạng có thể mở, kích thước và nhãn hợp lệ. Báo lỗi rõ ràng nếu thiếu file/nhãn.
- Giữ nguyên ảnh gốc; nếu cần resize/crop/chuẩn hóa, ghi kích thước, thứ tự kênh, miền giá trị và lý do.
- Áp dụng cùng preprocessing khi huấn luyện, đánh giá và dự đoán ảnh mới.

## Bước 4 — Dựng pipeline tối thiểu

1. Nạp và kiểm tra dữ liệu.
2. Tách train/validation/test phù hợp; ghi cách chia và seed nếu dùng ngẫu nhiên.
3. Fit preprocessing và mô hình chỉ trên train; chọn tham số bằng validation.
4. Chạy test một lần theo thiết kế và ghi chỉ số.
5. Chạy 5–10 ảnh demo độc lập, ghi dự đoán thực tế từng ảnh.
6. Lưu/hiển thị ảnh gốc cùng dự đoán; khi có nhãn đã xác nhận, hiển thị cả nhãn đúng.

Nếu dùng mô hình huấn luyện sẵn, thay bước huấn luyện bằng việc ghi chính xác nguồn/phiên bản, nhãn hỗ trợ, preprocessing, cách nạp và giới hạn. Không khẳng định mô hình nhận lớp mà nó không hỗ trợ.

## Bước 5 — Kiểm thử và phân tích lỗi

Điền [mẫu kiểm thử](./testing-template.md) sau khi chạy thật. Bao gồm cả ca dễ và khó; không bỏ các dự đoán sai. Với mỗi lỗi, kiểm tra ảnh, nhãn, preprocessing, lớp nhầm, chất lượng dữ liệu, ngưỡng/độ tin cậy nếu có, rồi ghi nguyên nhân có bằng chứng hoặc ghi `[GIẢ THUYẾT — CẦN KIỂM CHỨNG]`.

Phân biệt:

- **Đánh giá test set:** số liệu trên tập được giữ riêng theo thiết kế.
- **Ảnh demo 5–10 mẫu:** minh họa từng đầu vào/đầu ra, không đủ để đại diện cho mọi trường hợp.
- **Ví dụ chưa chạy:** chỉ là kịch bản dự kiến, không ghi như kết quả.

## Bước 6 — Trình bày và tái lập

Demo nên cho người xem thấy đầu vào, các bước chính, nhãn dự đoán và lỗi/giới hạn. Kèm lệnh chạy, phiên bản môi trường, cấu trúc dữ liệu cần thiết và cách tái lập. Chỉ công bố hình/bảng đã tạo từ lần chạy có thật; ẩn thông tin cá nhân khi cần.

## Checklist hoàn tất

- [ ] Bài toán, đầu vào, đầu ra, người dùng và phạm vi rõ.
- [ ] Nguồn/quyền dùng dữ liệu và nhãn được ghi lại.
- [ ] Pipeline train/inference nhất quán, không rò rỉ test.
- [ ] Mô hình, phiên bản, tham số và môi trường được nêu.
- [ ] Có tối thiểu 5 ảnh kiểm thử thực tế, ưu tiên đủ 5–10 ảnh và có nhãn đúng đã xác nhận.
- [ ] Kết quả đúng/sai, trường hợp khó và phân tích lỗi được ghi trung thực.
- [ ] Demo chạy lại được; số liệu và hình minh họa truy nguyên được.
- [ ] Hạn chế/rủi ro và nội dung cần sinh viên tự xác nhận được ghi rõ.
