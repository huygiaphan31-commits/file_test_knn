# Đánh giá preset B với k=1000

## Thay đổi cấu hình

- Preset B dùng tối đa 500 mẫu/lớp và cố định `k=1000`.
- Preset A giữ nguyên: tối đa 200 mẫu/lớp, chọn `k` trong
  `{1,3,5,7,9}` bằng validation.
- KNN tự cài, khoảng cách Euclid, bỏ phiếu đa số và quy tắc phá hòa không đổi.
- Nếu train indices sau khi tách validation có ít hơn 1000 mẫu, chương trình
  báo `ValueError` và không tự giảm `k`. Ở `ManualKNN`, `k` lớn hơn số mẫu
  thực tế cũng bị từ chối rõ ràng.

## So sánh ghép cặp trên validation

Chạy một lần với 31.000 mẫu train EMNIST ByClass, cùng preprocessing hiện tại,
cùng seed 42, cùng cách chia validation như `fit_classifier` của ứng dụng:
620 mẫu validation (10/lớp), 30.380 mẫu dùng để tìm láng giềng trong validation.
Chỉ mở official train IDX và mapping; không mở test IDX.

| k | Validation macro-F1 |
|---:|---:|
| 9 | 0.591435475986 |
| 1000 | 0.398248683684 |

Macro-F1 tại k=1000 giảm **0.193186792302** so với k=9 trên đúng phép thử này.
Đây là phép so sánh validation, không phải test score. Preset B vẫn cố định
k=1000 theo yêu cầu; không tự chuyển về k=9.

Thời gian tìm láng giềng và chấm **cả hai k cùng một lượt** là 37.511 giây.
Đây là thời gian scoring ghép cặp (cùng tìm đến max k=1000), không phải latency
riêng cho từng k. Nạp dữ liệu mất 4.077 giây, preprocessing 4.842 giây, tổng
lượt benchmark 46.453 giây trên máy chạy hiện tại.

Artefact chi tiết:

- [`summary.json`](./run-20261024-120148/summary.json)
- [`validation_scores.csv`](./run-20261024-120148/validation_scores.csv)

Benchmark lịch sử khác với split riêng ghi k=9, macro-F1 0.614767;
không dùng score đó để so sánh trực tiếp với lượt này. Các artefact cũ không
bị ghi đè.

## Kiểm tra giá trị app

Gọi loader thật cho cấu hình B trả:

- mode: `fixed_1000`
- train: 31.000
- validation: 620
- `selected_k`: **1000**
- validation macro-F1: `0.398248683684`
- probe trả về đúng 1000 láng giềng.

## Kiểm thử

Lệnh chạy từ thư mục demo:

```powershell
python -B -m unittest discover -s tests -t .
```

Kết quả: **62 tests đạt, 0 trượt**.

## Giới hạn

- k=1000 có macro-F1 thấp hơn đáng kể k=9 trên validation split này; preset B
  vẫn sử dụng 1000 theo yêu cầu thử nghiệm.
- Không đo trên official test và không dùng test để chọn tham số.
- Hiệu năng nhận diện ảnh ngoài không được đánh giá trong lượt benchmark này.
- Di chuyển ứng dụng đang chạy cần bấm **Làm mới model** hoặc khởi động lại
  để xóa model cache cũ và nạp cấu hình mới.
