# Báo cáo làm mới model và chẩn đoán pipeline

## 1. Mục tiêu và cấu hình

Kiểm tra luồng nạp model của giao diện Streamlit, xác nhận preset B trên dữ
liệu EMNIST cục bộ, rà soát phân đoạn/tiền xử lý/ánh xạ lớp và sửa hai lỗi đã
tái hiện trên ảnh được cung cấp.

Không thay thuật toán KNN tự cài, khoảng cách Euclid, bỏ phiếu, quy tắc hòa
phiếu, dữ liệu EMNIST hay hai preset. Thay đổi giới hạn ở phân đoạn và
preprocessing dùng chung; không mở official test IDX hay tải/xóa dữ liệu.

## 2. Chẩn đoán trực tiếp ảnh `aB3k9C5dA7`

Tệp `data/my_samples/aB3k9C5dA7.png` tồn tại, đọc được dưới dạng PNG RGBA,
kích thước 952×272. Ground truth `aB3k9C5dA7` do người dùng xác nhận; ảnh này
đã dùng trong chẩn đoán nên **không phải independent holdout**. Model chạy là
preset B: 31.000 train, k=9, validation macro-F1 `0.591435`.

### Phân đoạn trước/sau

Trước thay đổi, Otsu toàn ảnh tạo một foreground component lớn vì nền giấy
phía dưới tối dần. Box `(461,102)-(952,272)` nuốt nhiều ký tự đến mép phải và
chồng lấn box kế tiếp; 10 box đếm được không tương ứng với 10 ký tự phân đoạn
đúng. Mask, box đánh số và crop trước sửa ở
[`aB3k9C5dA7-diagnosis-20261024/`](./aB3k9C5dA7-diagnosis-20261024/).

Sau bù nền cục bộ, có đúng 10 box trái-sang-phải, lần lượt bao quanh
`a B 3 k 9 C 5 d A 7`; không còn box khổng lồ, vùng gộp hay ký tự bị bỏ.
Kết quả trực quan và crop ở
[`aB3k9C5dA7-diagnosis-20261024-after/`](./aB3k9C5dA7-diagnosis-20261024-after/).

### Phân loại từng vùng

| # | Box sau sửa (x0,y0,x1,y1) | Đúng | Trước sửa phân đoạn | Sau chỉ sửa phân đoạn | Sau cả hai sửa đổi |
|---:|---|:---:|:---:|:---:|:---:|
| 1 | (71,118,113,161) | a | j | j | J |
| 2 | (160,91,211,162) | B | j | j | B |
| 3 | (285,95,332,174) | 3 | j | j | j |
| 4 | (369,102,411,163) | k | Y | Y | k |
| 5 | (471,102,505,164) | 9 | i (box gộp) | j | 9 |
| 6 | (528,100,574,171) | C | j (box chồng lấn) | j | C |
| 7 | (613,111,652,173) | 5 | j | j | 5 |
| 8 | (711,102,747,179) | d | j | j | j |
| 9 | (781,103,829,177) | A | j (không có box riêng) | j | h |
| 10 | (864,104,897,180) | 7 | j | j | t |

Trước sửa, chuỗi vùng là `jjjYijjjjj`, nhưng sau box 5 các vùng bị gộp/chồng
lấn nên không thể coi đó là phép đo phân loại trên 10 crop đúng. Sau chỉ sửa
phân đoạn, chuỗi vùng là `jjjYjjjjjj`, **0/10**. Sau cả hai sửa đổi, chuỗi
vùng là `JBjk9C5jht`, **5/10**; chuỗi `recognize_line` hiển thị
`JB jk 9C5 jht` do khoảng cách giữa các vùng làm chèn dấu cách. Đây là kết quả
trên ảnh dùng chẩn đoán, không phải đánh giá tổng quát.

CSV có box, nhãn thật, dự đoán và 9 láng giềng/khoảng cách cho từng vùng:
[`03_region_predictions_and_neighbors.csv`](./aB3k9C5dA7-diagnosis-20261024-final/03_region_predictions_and_neighbors.csv).
Crop gốc và glyph 28×28 sau sửa ở
[`aB3k9C5dA7-diagnosis-20261024-final/`](./aB3k9C5dA7-diagnosis-20261024-final/).

## 3. Rà soát luồng ảnh và model

### 3.1 Ảnh train và ảnh tải lên

- `data_loader.load_emnist_training()` chỉ mở train images, train labels và
  mapping. Loader riêng `load_emnist_byclass()` mới mở test IDX; luồng nạp app
  không gọi loader này.
- `preprocessing.orient_emnist_images()` transpose batch EMNIST IDX từ
  `(N, 28, 28)`; ảnh ngoài được đọc grayscale, không trải qua phép transpose
  EMNIST.
- Cả glyph EMNIST đã định hướng và crop ảnh ngoài đều đi qua
  `normalize_glyph()`: xác định polarity theo median viền, trừ mức nền viền,
  tìm nét, crop, resize giữ tỷ lệ trong 20×20, căn giữa trên canvas 28×28,
  chuẩn hóa cực đại cường độ mỗi glyph về 255, rồi scale pixel về `[0,1]` và
  flatten thành vector 784 chiều.
- Trước chuẩn hóa max, query trên ảnh này có max `0.4745–0.5765`, trong khi
  99,6% vector train B có max 1,0 (phân vị thấp nhất 0.8745). Đây là bằng
  chứng trực tiếp về khác biệt cường độ giữa ảnh bút xanh/nền xám và train.
- Ảnh glyph 28×28 sau chuẩn hóa giữ nét và có max 1,0; chưa có dữ liệu độc lập
  để kết luận ảnh khác cũng được cải thiện.

Một thí nghiệm validation train-only đã chạy trước đó so sánh preprocessing
hiện tại với phép chia cho max áp dụng **đồng nhất cả train và validation**,
cùng split, classifier và k=9. Cả hai nhánh cho macro-F1
`0.5914354759860289`. Phép thử này không đánh giá biến đổi query-only trên ảnh
mới và không cung cấp bằng chứng tổng quát để tích hợp tăng tương phản.

### 3.2 Phân đoạn và phân loại

Với ảnh dòng, `line_recognition.segment_line()` tạo foreground mask, tìm
connected components, ghép chấm rời, tách valley rõ và sắp trái-sang-phải.
Lỗi tái hiện trên ảnh này là ngưỡng Otsu toàn ảnh nhầm vùng nền giấy tối dần
thành foreground, tạo box khổng lồ. Sửa tối thiểu: bù nền Gaussian-blur cục
bộ (bán kính 4% cạnh ngắn nhất) trước Otsu; nếu residual tối đa không vượt 8
thì trả mask rỗng để nền xám phẳng không bị coi là nét. Không đổi logic
connected components hay tách/ghép.

Đã thêm kiểm thử hồi quy cho nền giấy gradient và ảnh nền xám phẳng. Trên ảnh
đích, box sau sửa khớp cả 10 ký tự theo quan sát. Đây là xác nhận lỗi phân
đoạn trên ảnh này, không khẳng định mọi kiểu nền/ký tự dính đều được xử lý.

Với 10 vùng đã phân đoạn đúng, preprocessing cũ vẫn sai 10/10. Đo được khác
biệt cường độ query/train nêu trên. Chuẩn hóa max nhất quán trong
`normalize_glyph()` hiện dùng chung cho train EMNIST và ảnh ngoài. Trên ảnh
chẩn đoán, giữ nguyên model/k/box và dùng cùng cách biểu diễn cho hai phía
cho kết quả 5/10; các lớp còn nhầm là `a→J`, `3→j`, `d→j`, `A→h`, `7→t`.
Láng giềng cho thấy còn nhập nhằng các nhóm Q/C/G, j, j/d, A/h và 7/t.

### 3.3 Ánh xạ nhãn

Mapping EMNIST cục bộ có 62 class ID và 62 ký tự duy nhất. Các ranh giới được
xác nhận trực tiếp: `0→0`, `9→9`, `10→A`, `35→Z`, `36→a`, `61→z`.
`load_character_mapping()` kiểm tra đầy đủ ID 0–61 và tập ký tự hợp lệ; test
mới bao phủ các ranh giới chữ số/chữ hoa/chữ thường. Không tìm thấy bằng chứng
mapping là nguyên nhân của hiện tượng trên ảnh không có.

## 4. Làm mới cache và nạp model

`load_demo_model()` được cache bằng `st.cache_resource`; cấu hình truyền vào
cache gồm thư mục dữ liệu, giới hạn train/lớp và chế độ chọn k. Đã bổ sung nút
**Làm mới model** riêng với **Nạp model**:

1. Nút làm mới tăng generation dùng chung và clear cache của `load_demo_model`.
2. Session đang yêu cầu refresh xóa model/kết quả cũ, sau đó nạp lại train và
   tạo `ManualKNN` đồng bộ.
3. Generation được lưu cùng model trong session; session khác sẽ loại model
   cũ và kết quả dự đoán ở lần rerun kế tiếp.
4. Spinner thể hiện đang nạp; lỗi đọc/kiểm tra dữ liệu được hiển thị rõ và
   không giữ model session cũ.
5. Nút **Nạp model** thông thường vẫn tái sử dụng cache, tránh nạp lại khi chỉ
   tải ảnh hoặc nhận diện.

Đã gọi trực tiếp refresh cho preset B trên dữ liệu đang có, chỉ đọc train và
mapping:

| Số mẫu train | Validation | k | Validation macro-F1 |
|---:|---:|---:|---:|
| 31.000 | 620 | 9 | 0,591435475986 |

Đây là validation score của lần nạp app hiện tại, không phải test accuracy.
Artefact benchmark cố định khác trong
[`train-size-validation-20261022-163943/summary.json`](./train-size-validation-20261022-163943/summary.json)
báo B: 31.000 mẫu, k=9, macro-F1 0,614767 trên protocol/split riêng; không
so sánh hai score như cùng một split. Preset A vẫn mặc định 200 ảnh/lớp và
chọn k bằng validation; artefact benchmark chung trước đó ghi 12.400 mẫu,
k=7, macro-F1 0,568428.

Để làm mới trên giao diện: chạy app, chọn cấu hình mong muốn trong sidebar,
sau đó bấm **Làm mới model**. Khi dữ liệu đầy đủ, B phải hiển thị 31.000 mẫu,
k=9 và validation macro-F1 của lần nạp đó. Chọn **Nạp model** chỉ dùng cache
nếu cấu hình đã có.

## 5. Kết quả ảnh ngoài và giới hạn

Artefact trước đó ghi nhận trên `eight-characters.png` (pilot đã dùng chọn
phép thử): pipeline hiện tại đúng 0/8, biến thể scale query-only đúng 5/8.
Trên `sample-01.png`, hai nhánh đều đúng 0/6 theo nhãn đã xác nhận trước.
`sample-02.png` không có ground truth xác nhận và không được tính điểm. Ảnh
`aB3k9C5dA7.png` cũng vừa được dùng để chẩn đoán/sửa; vì vậy hiện có **0 ảnh**
đủ điều kiện `independent_holdout`. Những số liệu này không chứng minh cải
thiện tổng quát.

Đã tích hợp hai sửa đổi có bằng chứng cụ thể: bù nền cục bộ cho lỗi phân đoạn
trên nền gradient và chuẩn hóa max glyph dùng chung để khớp cường độ query
với train. Validation train-only B sau chuẩn hóa vẫn macro-F1 `0.591435` trên
cùng protocol; trên ảnh chẩn đoán, số đúng đổi từ 0/10 sau sửa phân đoạn riêng
sang 5/10 sau cả hai sửa đổi. Không dùng official test IDX. Năm ký tự còn sai
chưa được tinh chỉnh theo ảnh này; cần ảnh mới với ground truth xác nhận trước
khi dự đoán để đánh giá độc lập.

## 6. Kiểm thử và khởi chạy

Lệnh toàn bộ unit tests (chạy từ thư mục demo):

```powershell
python -B -m unittest discover -s tests -t .
```

Kết quả lần chạy cuối: **60 tests, 60 đạt, 0 trượt**.

Smoke test ứng dụng:

```powershell
python -B -m streamlit run app.py --server.headless true --server.port 8765
```

Sau các sửa đổi hiện tại, endpoint
`http://localhost:8766/_stcore/health` trả HTTP 200. Tiến trình smoke test đã
được dừng sau khi kiểm tra.

Chạy giao diện tương tác:

```powershell
streamlit run app.py
```

## 7. Tệp đã sửa/tạo

### Sửa trong lượt chẩn đoán ảnh này

- `line_recognition.py` — bù nền cục bộ trước Otsu.
- `preprocessing.py` — chuẩn hóa cường độ max glyph dùng chung cho train/query.
- `tests/test_line_recognition.py` — test nền gradient, nền xám phẳng và 10
  vùng khoảng cách không đều; fixture vẽ đủ 10 vùng.
- `tests/test_preprocessing.py` — test chuẩn hóa glyph nét có cường độ thấp.
- `results/final_model_and_pipeline_report.md` — báo cáo này.

### Thay đổi refresh model từ lượt trước, vẫn có trong workspace

- `app.py` — nút refresh cache, generation để vô hiệu model session cũ, trạng
  thái đang nạp/thành công/lỗi.
- `tests/test_app.py` — test refresh cache và cấu hình preset A/B.
- `tests/test_data_loader.py` — test mapping tại các class ID ranh giới.

### Artefact chẩn đoán được tạo mới

- `results/aB3k9C5dA7-diagnosis-20261024/` — mask, box/crop và kết quả trước
  sửa.
- `results/aB3k9C5dA7-diagnosis-20261024-after/` — sau sửa phân đoạn, trước
  chuẩn hóa cường độ.
- `results/aB3k9C5dA7-diagnosis-20261024-final/` — sau cả hai sửa đổi, gồm
  ảnh 28×28 và láng giềng.

Không sửa `knn.py`, cấu hình, dữ liệu EMNIST hoặc tệp ngoài thư mục demo.
Không ghi đè artefact thí nghiệm cũ. Model đã cache trong app đang chạy cần
bấm **Làm mới model** hoặc khởi động app mới để dựng lại vector train theo
preprocessing mới.
