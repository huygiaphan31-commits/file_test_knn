# Demo nhận diện chữ số viết tay bằng KNN

Chương trình minh họa một bài toán Computer Vision bằng Python và scikit-learn. Chương trình dùng `sklearn.datasets.load_digits()`, trong đó mỗi ảnh chữ số có kích thước **8×8 pixel**. Đây **không phải MNIST 28×28**.

## Cài đặt

Khuyến nghị tạo môi trường ảo:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Nếu PowerShell chặn kích hoạt môi trường, có thể dùng trực tiếp:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Chạy

Từ thư mục dự án:

```powershell
python demo.py
```

Mặc định chương trình dùng `k=3`, chọn 8 ảnh kiểm thử và lưu các kết quả thực tế vào `results/`. Để mở biểu đồ tương tác:

```powershell
python demo.py --show
```

Có thể đổi số láng giềng hoặc số ảnh mẫu (5–10):

```powershell
python demo.py --k 5 --samples 10
```

Tham số `--samples` ngoài khoảng 5–10 hoặc `--k` không hợp lệ sẽ báo lỗi. Có thể xem các tùy chọn bằng `python demo.py --help`.

## Pipeline

1. Nạp ảnh và nhãn từ `load_digits()`.
2. Làm phẳng mỗi ảnh 8×8 thành vector 64 giá trị pixel để KNN so sánh.
3. Chia dữ liệu thành train/test theo tỷ lệ 75/25, có phân tầng theo nhãn và `random_state=42` để tái lập.
4. Huấn luyện `KNeighborsClassifier(n_neighbors=3)`.
5. Dự đoán tập test; tính accuracy và confusion matrix.
6. Lấy 5–10 ảnh từ tập test có nhãn, ưu tiên có cả dự đoán đúng và sai nếu có; hiển thị nhãn thật, nhãn dự đoán và trạng thái.
7. Lưu biểu đồ cùng số liệu đã tính trong lần chạy hiện tại.

`k=3` là giá trị nhỏ, dễ giải thích cho người mới và phù hợp với ví dụ KNN trong báo cáo mẫu của dự án. Giá trị lẻ cũng giúp giảm khả năng hòa phiếu trong phân loại hai lớp, dù bài toán chữ số có nhiều lớp và không loại bỏ mọi trường hợp hòa. Có thể thử `--k 5` để so sánh; không nên khẳng định giá trị nào tốt hơn trước khi chạy và đánh giá trên cùng cách chia dữ liệu.

Mặc định KNN của scikit-learn dùng khoảng cách Minkowski với `p=2` (Euclidean). Chương trình không chuẩn hóa thêm pixel: nó sử dụng trực tiếp biểu diễn của dataset.

## File kết quả

Sau khi chạy thành công, thư mục `results/` chứa:

- `confusion_matrix.png`: hàng là nhãn thật, cột là nhãn dự đoán.
- `test_samples.png`: ảnh mẫu, nhãn thật/dự đoán và trạng thái đúng/sai.
- `metrics.json`: cấu hình chia dữ liệu, accuracy, ma trận nhầm lẫn và kết quả của từng ảnh mẫu.

Các file số liệu và biểu đồ được tạo khi chương trình thực sự chạy; không cung cấp trước số liệu minh họa. Những ảnh kiểm thử riêng này được lấy từ tập test có sẵn nhãn của `load_digits()`, **không phải ảnh tự chụp**. Bảng 5–10 ảnh giúp trình bày từng trường hợp, nhưng không thay thế đánh giá trên toàn bộ tập test và không đảm bảo đại diện cho ảnh thực tế ngoài dataset.

## Giới hạn

- `load_digits()` là dataset ảnh chữ số nhỏ 8×8; kết quả không tự động áp dụng cho ảnh chụp chữ số ngoài đời hoặc dữ liệu MNIST 28×28.
- KNN dựa vào khoảng cách giữa vector pixel; thay đổi vị trí, nền, kích thước hay độ dày nét có thể ảnh hưởng dự đoán.
- Accuracy chỉ phản ánh tập test được dùng trong lần chạy này. Không xem 5–10 ảnh mẫu là bằng chứng hiệu năng tổng quát.
- Chương trình không huấn luyện CNN và không nhận diện nhiều chữ số trong cùng một ảnh.

Khi đưa kết quả vào báo cáo, ghi rõ dataset, cách chia, tham số, kết quả thực tế, ảnh kiểm thử và các trường hợp sai; không dùng số liệu trong báo cáo mẫu như kết quả của lần chạy này.
