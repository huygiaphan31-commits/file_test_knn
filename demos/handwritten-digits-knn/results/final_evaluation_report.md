# Final evaluation: handwritten-character KNN demo

## 1. Mục tiêu và cấu hình

Đối chiếu hai cách chuẩn bị vector ảnh ngoài trước khi đưa vào cùng classifier
KNN hiện có:

- **Hiện tại:** preprocessing glyph đang dùng trong web, trả vector 28×28 = 784
  chiều.
- **Tăng tương phản thử nghiệm:** sau preprocessing hiện tại, chia vector query
  cho giá trị lớn nhất của chính vector đó. Vector train không thay đổi.

Hai nhánh dùng cùng ảnh, cùng các bounding box lấy một lần, cùng thứ tự ký tự,
cùng model B nạp từ EMNIST train (31.000 mẫu) và `k=9`. Đây là thử nghiệm riêng;
không tích hợp phép biến đổi vào app.

EMNIST official test IDX không được mở. Không tải hoặc xóa dữ liệu.

## 2. Phương pháp và nguồn nhãn

Manifest:
[`labeled_evaluation_manifest.csv`](./eight-characters-diagnosis/labeled_evaluation_manifest.csv)

| Ảnh | Ground truth | Nguồn | Vai trò |
|---|---|---|---|
| `data/my_samples/eight-characters.png` | `aAbBcCdD` | Người dùng xác nhận trong hội thoại | `pilot`: đã dùng để lựa chọn phép thử tăng tương phản; không độc lập |
| `data/my_samples/sample-01.png` | `Ketqua` (khoảng trắng giữa từ được bỏ khi ghép các vùng) | Artefact trước ghi nhận “Ket qua” đã được người dùng xác nhận; các nhãn vùng được ghi trong chẩn đoán trước | `previously_used_comparison`: đã được chẩn đoán trước; không phải holdout mù |
| `data/my_samples/sample-02.png` | Chưa có | Không có nhãn xác nhận | Loại khỏi tính điểm |

`eight-characters.png` phân đoạn thành 8 vùng; `sample-01.png` thành 6 vùng.
Trên mỗi ảnh, hai nhánh nhận cùng crop/padding và cùng bounding box. Macro-F1
trong báo cáo được tính trên hợp các lớp xuất hiện trong nhãn thật **hoặc**
dự đoán, nên các dự đoán sai vào lớp ngoài ground truth được tính là false
positive. Với số mẫu rất nhỏ, macro-F1 chỉ mô tả các ảnh này.

## 3. Preset A/B: validation chung đã có

Không chạy lại benchmark tốn thời gian. Dùng artefact
[`train-size-validation-20261022-163943/summary.json`](./train-size-validation-20261022-163943/summary.json)
và [`validation_scores.csv`](./train-size-validation-20261022-163943/validation_scores.csv).
Artefact mô tả một validation cố định, phân tầng 10 mẫu/lớp (620 mẫu), seed 42,
được dùng chung cho các giới hạn train; chỉ đọc EMNIST train và mapping, không
đọc test IDX.

| Preset tương ứng | Train | k | Validation macro-F1 trên split chung | Thời gian scoring đã đo |
|---|---:|---:|---:|---:|
| A — 200/lớp, chọn k trên validation | 12.400 | 7 | 0,568428 | 16,50 giây |
| B — 500/lớp, cố định k=9 | 31.000 | 9 | 0,614767 | 35,09 giây |

Scoring time là một lượt validation tính đồng thời mọi candidate k cho giới hạn
train tương ứng, không phải latency của một ảnh. Artefact ghi thời gian nạp
pool chung 500/lớp là 3,13 giây và preprocessing chung là 2,79 giây; không tách
thành thời gian nạp riêng cho từng preset. Trong cùng artefact, thời gian dự
đoán ảnh `sample-01.png` được ghi là 0,242 giây (A) và 0,462 giây (B); đây chỉ
là một ảnh và phụ thuộc máy chạy.

**Lưu ý về protocol:** các kết quả validation lịch sử A=0,588696 (k=3) và B
k=9=0,591435 được tạo từ cách lấy mẫu/split khác. Lần nạp model B trong script
đánh giá hiện tại cũng báo 0,591435. Không so sánh trực tiếp các giá trị đó với
benchmark split chung trên như thể cùng một split. Validation score không phải
độ chính xác trên ảnh ngoài hay test accuracy. Kết quả trên ảnh người dùng phải
đọc riêng ở mục sau.

## 4. So sánh hai cách preprocessing trên ảnh có nhãn

Model của lần chạy đánh giá: 31.000 mẫu train, `k=9`, validation macro-F1
0,591435. Số này được ghi từ lần nạp model đó; không dùng test IDX.

| Ảnh | Vai trò | Hiện tại: đúng/tổng, accuracy, macro-F1 | Tăng tương phản: đúng/tổng, accuracy, macro-F1 |
|---|---|---|---|
| `eight-characters.png` | Pilot | 0/8, 0%, 0,000 | 5/8, 62,5%, 0,455 |
| `sample-01.png` | Đã dùng trong chẩn đoán trước | 0/6, 0%, 0,000 | 0/6, 0%, 0,000 |
| Gộp mô tả (không phải holdout) | Hai vai trò trên | 0/14, 0%, 0,000 | 5/14, 35,7%, 0,246 |

Macro-F1 của pilot ở artefact thử nghiệm trước từng được báo 0,625 vì chỉ lấy
trung bình trên 8 lớp ground truth; cách tính đó bỏ khỏi trung bình các lớp
được dự đoán sai nhưng không có trong ground truth (`Q`, `l`, `j`). Báo cáo này
dùng hợp nhãn thật/dự đoán; theo cách tính nhất quán đó, macro-F1 pilot là
0,455. Accuracy và số ký tự đúng không đổi.

### Lỗi theo ký tự

- **Pilot, hiện tại:** `a,A,b,B,c,C` bị đoán thành `i`; `d` thành `j`.
- **Pilot, tăng tương phản:** đúng `b,B,c,C,D`; `a→Q`, `A→l`, `d→j`.
- **sample-01, hiện tại:** `K→k`, `e→c`, `t→l`, `q→i`, `u→w`, `a→0`.
- **sample-01, tăng tương phản:** `K→k`, `e→c`, `t→l`, `q→l`, `u→w`, `a→0`.

Confusion matrix đầy đủ và per-class counts nằm trong
[`confusion_matrices.csv`](./eight-characters-diagnosis/labeled_contrast_evaluation/run-20261024-110443-598851/confusion_matrices.csv);
từng bounding box, 9 láng giềng và khoảng cách nằm trong
[`per_character.csv`](./eight-characters-diagnosis/labeled_contrast_evaluation/run-20261024-110443-598851/per_character.csv).
Ảnh đối chiếu preprocessing:
[pilot](./eight-characters-diagnosis/labeled_contrast_evaluation/run-20261024-110443-598851/image-01-eight-characters/comparison.png),
[sample-01](./eight-characters-diagnosis/labeled_contrast_evaluation/run-20261024-110443-598851/image-02-sample-01/comparison.png).

## 5. Đánh giá độc lập và giới hạn

Hiện có **0 ảnh** đủ điều kiện `independent_holdout`. Ảnh pilot đã tham gia
lựa chọn phương pháp; `sample-01.png` đã có chẩn đoán trước; `sample-02.png`
không có nhãn xác nhận. Vì vậy số 5/8 không phải kết quả độc lập, và không
được kết luận tăng tương phản cải thiện tổng quát. Gộp 14 ký tự ở trên chỉ là
tóm tắt hai ảnh đã biết, không phải đánh giá holdout.

Các giới hạn khác:

- Mỗi lớp trong ảnh pilot chỉ xuất hiện một lần; macro-F1 rất nhạy với một lỗi.
- `sample-01.png` là một câu ngắn đã được xem trước đó, không đại diện các kiểu
  chữ viết khác.
- `sample-02.png` không thể tính accuracy/confusion matrix vì thiếu nhãn thật.
- Phép chia theo max làm thay đổi cường độ query so với train, nhưng chưa có
  dữ liệu độc lập để xác định lợi ích/rủi ro theo người viết, ánh sáng, bút và
  nền khác nhau.

## 6. Quyết định tích hợp

**Giữ nguyên pipeline web; chưa tích hợp tăng tương phản.** Có tín hiệu tốt trên
ảnh pilot nhưng không cải thiện trên ảnh đã chẩn đoán trước, và chưa có ảnh
independent holdout hợp lệ. Chưa đủ bằng chứng để chọn cách mới cho người dùng
chung.

Không sửa `app.py`, `preprocessing.py`, `knn.py`, cấu hình model hoặc thuật toán
KNN trong lượt này.

## 7. Tái hiện và bổ sung ảnh độc lập

Từ thư mục `demos/handwritten-digits-knn/`, chạy:

```powershell
python -B results\eight-characters-diagnosis\evaluate_labeled_contrast.py
```

Script xác thực manifest và đường dẫn ảnh, kiểm tra số ký tự ground truth khớp
số vùng trước khi nạp model, và ghi mỗi lần chạy vào `run-*` mới để không ghi
đè kết quả cũ. Thêm một hàng trong manifest theo mẫu
[`labeled_evaluation_manifest_template.csv`](./eight-characters-diagnosis/labeled_evaluation_manifest_template.csv):

- Đặt ảnh mới dưới `data/my_samples/` và ghi đường dẫn tương đối trong demo.
- Ghi chuỗi ký tự đúng theo thứ tự trái sang phải, bỏ dấu cách giữa các từ.
- Xác nhận ground truth **trước khi xem dự đoán**; ghi nguồn xác nhận.
- Chỉ đánh dấu `independent_holdout` khi `label_confirmed_before_prediction=yes`,
  `used_for_method_selection=no`, `used_in_prior_diagnostics=no`.
- Giữ ảnh holdout không dùng để thay đổi phép preprocessing hay tham số; nếu
  chỉnh phương pháp, cần một holdout mới cho lần đánh giá tiếp theo.

Một ảnh độc lập có nhãn là mức tối thiểu để script tính kết quả holdout, nhưng
không đủ cho kết luận tổng quát. Nên bổ sung nhiều ảnh mới từ nhiều lần viết/
người viết và lặp lại các lớp quan tâm; hướng dẫn thực hành là 5–10 ảnh, không
phải bảo đảm cỡ mẫu thống kê.

Artefact báo cáo của lần chạy hiện tại:
[`report.json`](./eight-characters-diagnosis/labeled_contrast_evaluation/run-20261024-110443-598851/report.json).

## 8. Kiểm thử

Lệnh đã chạy từ thư mục demo:

```powershell
python -B -m unittest discover -s tests -t .
```

Kết quả cuối: **54 tests, OK; 0 lỗi, 0 thất bại** (51 test hiện có và 3 test
mới cho macro-F1/confusion, kiểm tra holdout và xác thực đường dẫn/nhãn).

## 9. Tệp tạo/sửa

Đã sửa:

- `README.md` — hướng dẫn chạy đánh giá preprocessing và bổ sung nhãn.
- `results/eight-characters-diagnosis/evaluate_labeled_contrast.py` — kiểm tra
  manifest/holdout, macro-F1 trên hợp nhãn, confusion matrix và output run riêng
  để không ghi đè.
- `results/eight-characters-diagnosis/labeled_evaluation_manifest.csv` — thêm
  cột ghi nhận nguồn xác nhận và điều kiện độc lập.
- `results/eight-characters-diagnosis/labeled_evaluation_manifest_template.csv`
  — thêm các cột điều kiện holdout.

Đã tạo:

- `tests/test_contrast_evaluation.py`.
- Báo cáo này.
- Artefact đánh giá mới tại
  `results/eight-characters-diagnosis/labeled_contrast_evaluation/run-20261024-110443-598851/`
  (JSON, per-character CSV, confusion matrix, ảnh crop/28×28 và contact sheet).

Các script/artefact thử nghiệm pilot trước đó được giữ nguyên. Không có file
ngoài `demos/handwritten-digits-knn/` bị sửa.
