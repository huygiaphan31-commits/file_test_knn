# Nhận diện chữ viết tay bằng k-NN tự cài đặt

Demo phân loại ký tự thuộc 62 lớp EMNIST ByClass (0–9, A–Z, a–z). Có thể đưa
nhiều ảnh trong một lần chạy; mỗi ảnh chứa một ký tự hoặc một dòng ký tự. Với
ảnh dòng, chương trình dùng connected components để tách ký tự từ trái sang
phải, rồi ghép dự đoán thành chuỗi. Dấu tiếng Việt không có trong bộ lớp này.

## Cơ sở và giả định

- **[TÀI LIỆU NGUỒN]** Tài liệu đồ án của dự án trình bày nhận diện chữ số bằng
  KNN và khoảng cách Euclid. Xem
  [tóm tắt tài liệu nguồn](../../.github/skills/computer-vision-project/references/source-summary.md).
  Theo công thức người dùng cung cấp từ giáo trình trang 144:
  `d(x_i,x_j)=sqrt(sum_l((a_l(x_i)-a_l(x_j))^2))`.
- **[KIẾN THỨC BỔ SUNG]** EMNIST ByClass cung cấp 62 lớp ký tự chữ số và Latin
  hoa/thường, ảnh 28×28. Trang dataset NIST:
  <https://www.nist.gov/itl/products-and-services/emnist-dataset>. Hãy xác minh
  điều khoản dataset hiện hành trước khi dùng hoặc phân phối dữ liệu.
- **[GIẢ ĐỊNH ĐÃ DUYỆT]** Ảnh ngoài chứa một ký tự hoặc một dòng ký tự. CLI dùng
  subset có seed cố định, tối đa mặc định 200 ảnh train và 20 ảnh test mỗi lớp;
  giao diện web dùng duy nhất 500 ảnh train mỗi lớp. Các giới hạn nhằm giảm thời
  gian tính KNN; đây không phải toàn bộ benchmark.

Chương trình không dùng `KNeighborsClassifier`, scikit-learn, SciPy hay deep
learning. Các bước tính khoảng cách, sắp xếp láng giềng, bỏ phiếu và phá hòa do
module `knn.py` tự thực hiện. NumPy chỉ dùng để lưu vector và thực hiện phép
tính Euclid hàng loạt theo cùng công thức.

## Cài đặt

Tương thích cú pháp với Python 3.13.14; dependencies là NumPy, Pillow và
Matplotlib. Chưa cần cài đặt lại nếu chúng đã có trong môi trường:

```powershell
cd D:\Computer-Vision-Project\demos\handwritten-digits-knn
python -m pip install -r requirements.txt
```

## Dữ liệu

Tải ByClass từ trang NIST nêu trên sau khi đọc điều khoản, rồi giải nén theo
hướng dẫn trong [data/README.md](data/README.md). Chương trình **không tự tải
dataset**. Loader đọc IDX gzip theo luồng, dùng file mapping để ánh xạ class ID
sang ký tự, và kiểm tra magic number, số lượng mẫu, kích thước 28×28 và nhãn.

Nếu archive được giải nén vào vị trí khác, truyền `--data-dir`. Có thể chọn
subset nhỏ để chạy nhanh:

```powershell
python main.py --data-dir .\data\EMNIST --max-train-per-class 100 --max-test-per-class 10
```

Mặc định chạy đủ 62 lớp. Có thể giới hạn lớp để thử luồng phần mềm trước:

```powershell
python main.py --data-dir .\data\EMNIST --classes 0,1,10,36
```

Class ID luôn được đổi thành ký tự bằng mapping đi kèm, không tự giả định
class ID là mã ASCII.

## Pipeline và thuật toán

1. Đọc train và test IDX riêng biệt; lấy mẫu tối đa theo lớp với seed `42`.
2. Chuyển ảnh EMNIST về chiều hiển thị chuẩn bằng transpose. Sau đó EMNIST và
   crop bên ngoài cùng dùng chuẩn hóa glyph: xác định polarity/nét, cắt vùng
   nét, resize giữ tỉ lệ trong 20×20, căn giữa trên canvas 28×28 nền tối, scale
   pixel về `[0,1]` và làm phẳng thành vector 784 thuộc tính.
3. CLI chia train/validation theo lớp và chọn `k` trong `{1,3,5,7,9}` theo
   macro-F1; nếu đồng điểm, chọn `k` nhỏ hơn. Giao diện web dùng duy nhất cấu
   hình 500 ảnh/lớp (31.000 mẫu) và cố định `k=101`; nếu phần train sau khi giữ
   validation không đủ mẫu, chương trình báo lỗi thay vì tự giảm `k`.
4. Tính khoảng cách Euclid từ query tới từng vector train, sắp xếp tăng dần,
   lấy `k` láng giềng và chọn nhãn nhiều phiếu nhất. Nếu hòa, chọn nhãn có tổng
   khoảng cách láng giềng nhỏ hơn; nếu vẫn hòa, chọn theo thứ tự ký tự cố định
   lấy từ mapping.
5. Sau khi chọn `k`, dùng toàn bộ subset train đã lấy để đánh giá một lần trên
   official test subset. Báo cáo accuracy, macro-F1 và confusion matrix.
6. Ảnh đầu vào bên ngoài được chuyển grayscale trước chuẩn hóa; ảnh EMNIST giữ
   nguyên polarity IDX sau transpose rồi cũng qua cùng bước xác định nét, crop,
   resize và căn giữa. Không có nhãn thật cho ảnh ngoài trừ khi người dùng cung
   cấp nhãn riêng.
7. Với ảnh dòng, Otsu threshold và connected components xác định vùng nét;
   các chấm nhỏ gần phía trên thân chữ có thể được ghép vào đó. Nhiễu nhỏ được
   lọc theo kích thước/diện tích tương đối. Các vạch rất mảnh lặp lại sát cùng
   một mép ảnh chỉ bị loại khi tạo thành mẫu lặp trải dọc ảnh; một nét đơn lẻ
   sát mép được giữ lại. Vùng rộng chỉ được tách khi có khe dọc đủ rõ; khoảng
   cách lớn giữa vùng kế tiếp có thể thành dấu cách.

Khoảng cách dùng toàn bộ pixel thô sau scale, nên khác biệt về nét, căn chỉnh,
độ nghiêng, độ dày, ánh sáng hoặc phong cách viết có thể làm kết quả sai. Ảnh
đầu vào nên có nền tương đối đồng nhất và một dòng ngang. Ký tự dính nhau chỉ
được tách nếu có khe dọc hẹp; vùng không có khe đủ rõ sẽ được giữ nguyên.
Khoảng trắng được ước lượng bằng khoảng cách giữa hộp ký tự, chưa được kiểm
chứng trên nhiều kiểu viết. Dấu chấm và lọc nhiễu vẫn là heuristic. Nhiều dòng
chưa được nhận diện.

## Thử nghiệm so sánh tiền xử lý

Để so sánh pipeline trước đây với chuẩn hóa glyph dùng chung trên sáu crop và
một số ảnh train có nhãn, chạy:

```powershell
python -B compare_preprocessing.py
```

Script chỉ mở train IDX và mapping, dùng cùng subset train và cùng chỉ số
train/validation cho hai nhánh, rồi chọn `k` riêng bằng validation macro-F1.
Không mở test IDX. Kết quả được ghi vào thư mục mới
`results/preprocessing-comparison/` (hoặc thư mục hậu tố nếu đã tồn tại), gồm
ảnh ghép cặp `*_legacy.png`/`*_shared.png`, `crop_predictions.csv` và
`comparison.json`. Nhãn crop chỉ là dự đoán; không có accuracy crop vì nhãn
đúng của từng nét cắt chưa được xác minh.

## Chạy dự đoán một hoặc nhiều ảnh

Sau khi dữ liệu sẵn sàng:

```powershell
python main.py --data-dir .\data\EMNIST --predict-image D:\path\to\character.png
```

Tùy chọn nhận nhiều ảnh, mỗi ảnh có thể là một ký tự hoặc một dòng.
`--skip-evaluation` bỏ bước đọc/đánh giá test subset; chương trình vẫn dùng
train/validation để chọn `k` và tạo model một lần cho tất cả ảnh:

```powershell
python main.py --data-dir .\data\EMNIST --skip-evaluation --predict-images D:\images\line-01.png D:\images\line-02.jpg D:\images\line-03.png
```

Có thể truyền khoảng 5–10 đường dẫn theo cách tương tự. Nếu bỏ
`--skip-evaluation`, chế độ đánh giá hiện tại vẫn chạy trước rồi mới dự đoán.
Lệnh `--predict-image` cũ cũng tiếp tục được hỗ trợ.

Để kiểm tra quá trình phân đoạn và lưu ảnh trung gian vào thư mục riêng trong
`results/` mà không ghi đè kết quả đánh giá:

```powershell
python main.py --data-dir .\data\EMNIST --results-dir .\results\sample-01-debug-run --skip-evaluation --predict-image .\data\my_samples\sample-01.png --debug-segmentation
```

Lệnh trên giữ các kết quả đang có nguyên vẹn. Debug được lưu trong
`results/sample-01-debug-run/segmentation_debug/sample-01/` (hoặc thư mục có
hậu tố số nếu đã tồn tại), gồm `01_binary_mask.png`, `02_regions_numbered.png`
và `crop_01.png`, `crop_02.png`, ... cho từng vùng. Với debug prediction, file
`03_region_predictions.csv` ghi tọa độ, kích thước và nhãn dự đoán của từng vùng.

Hoặc từ thư mục gốc của dự án:

```powershell
python .\demos\handwritten-digits-knn\main.py --data-dir .\demos\handwritten-digits-knn\data\EMNIST --predict-image D:\path\to\character.png
```

Kết quả chuỗi của từng ảnh được in ra terminal; ảnh xem trước được lưu trong
`results/`. Ảnh ngoài không có ground truth tự động; không được xem dự đoán là
nhãn đúng. Chỉ dùng `--skip-evaluation` khi không cần số liệu đánh giá ở lần chạy.

## Giao diện web Streamlit

Cài các thư viện trong `requirements.txt`, sau đó khởi chạy từ thư mục demo:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Trong trình duyệt, giao diện chỉ dùng một cấu hình: tối đa 500 ảnh/lớp
(31.000 mẫu) và `k=101` cố định. Chọn chế độ một ký tự hoặc một dòng, tải ảnh
PNG/JPG, rồi bấm **Nạp model** (hoặc **Làm mới model** để xóa cache và đọc lại
dữ liệu train). Sau đó bấm **Nhận diện**. Với ảnh dòng, giao diện hiển thị vùng
đã đánh số và crop từng vùng. Nếu nội dung ảnh tải lên thay đổi, kết quả cũ
không còn khớp với ảnh hiện tại; cần bấm **Nhận diện** để tính lại. Không có
đánh giá test chạy tự động khi mở trang; validation macro-F1 chỉ mô tả tập
validation, không phải test accuracy hay bảo đảm hiệu năng trên ảnh ngoài.

### Đánh giá preprocessing ảnh ngoài

Thử nghiệm preprocessing hiện tại so với chuẩn hóa cường độ theo cực đại của
từng glyph được chạy tách biệt với pipeline web:

```powershell
python -B results\eight-characters-diagnosis\evaluate_labeled_contrast.py
```

Script dùng model B (31.000 mẫu train, `k=9`), phân đoạn từng ảnh một lần và
dùng lại chính các bounding box cho hai nhánh. Nó chỉ đọc EMNIST train để tạo
model; không mở test IDX. Kết quả mỗi lần chạy được tạo trong thư mục `run-*`
mới, không ghi đè kết quả cũ.

Manifest là
`results/eight-characters-diagnosis/labeled_evaluation_manifest.csv`. Mỗi hàng
cần đường dẫn ảnh tương đối trong demo, chuỗi ground truth theo thứ tự trái
sang phải (bỏ dấu cách giữa các từ), nguồn xác nhận nhãn và vai trò đánh giá.
Chỉ ghi `independent_holdout` khi ground truth đã được xác nhận trước khi xem
dự đoán, ảnh chưa dùng để chọn phương pháp và chưa dùng trong chẩn đoán trước.
Ba cột kiểm soát trong manifest ghi nhận các điều kiện đó; script từ chối
holdout nếu không thỏa mãn. Mẫu cột nằm trong
`results/eight-characters-diagnosis/labeled_evaluation_manifest_template.csv`.
Script đánh giá tương phản này giữ cấu hình `k=9` tường minh để tái lập phép
thử lịch sử; đó không phải giá trị của preset B hiện tại.

Không đưa ảnh không nhãn vào manifest tính điểm. Với mỗi ảnh, số ký tự nhãn
phải bằng số vùng phân đoạn. Báo cáo JSON/CSV gồm dự đoán, láng giềng, khoảng
cách, accuracy ký tự, macro-F1 mô tả và confusion matrix. Một ảnh nhỏ hoặc ảnh
đã dùng trong pilot không phải bằng chứng cải thiện tổng quát.

## Kiểm thử phần mềm và thực nghiệm

Chạy unit/integration tests:

```powershell
python -B -m unittest discover -s tests -t .
```

Tests tạo IDX gzip tổng hợp tạm thời để kiểm tra loader, vector Euclid, bỏ
phiếu, phá hòa, tiền xử lý và lệnh CLI. Đây chỉ là fixture kiểm thử phần mềm,
**không phải kết quả thực nghiệm EMNIST**.

Thực nghiệm trên EMNIST chỉ có sau khi tải dữ liệu và chạy demo. Chương trình
sinh kết quả trong `results/`:

- `metrics.json`: cấu hình lần chạy, số mẫu đã lấy, `k`, validation score,
  accuracy, macro-F1, confusion matrix và ví dụ.
- `test_predictions.csv`: từng ảnh test đã lấy, nhãn thật, nhãn dự đoán và đúng/sai.
- `confusion_matrix.csv`, `confusion_matrix.png`, `test_examples.png`.
- `query_prediction.png` cho một ảnh; `query_prediction_01.png`, ... cho nhiều ảnh.

Ít nhất 5 và mặc định 8 ví dụ được chọn từ tập test; nếu thực tế có dự đoán
đúng hoặc sai, hình sẽ có cả hai loại. Đây là ảnh thuộc test split của dataset,
không phải ảnh tự chụp. Để kiểm thử ảnh người dùng, chạy riêng từng ảnh có
`--predict-image`, ghi nhãn thật đã được người cung cấp xác nhận và nhận xét
vào báo cáo. Không gán nhãn thật theo dự đoán của mô hình.

## Cấu trúc module

| File | Vai trò |
|---|---|
| `main.py` | CLI, điều phối nạp dữ liệu, chọn `k`, đánh giá, lưu kết quả và dự đoán ảnh mới |
| `config.py` | Đường dẫn, kích thước ảnh, giới hạn mẫu, seed và `k` ứng viên |
| `data_loader.py` | Kiểm tra mapping và đọc IDX gzip theo luồng |
| `preprocessing.py` | Định hướng EMNIST, tạo vector 784 chiều, tiền xử lý ảnh ngoài |
| `knn.py` | Khoảng cách Euclid, tìm láng giềng, majority vote và phá hòa |
| `line_recognition.py` | Tách dòng bằng connected components, chuẩn hóa crop và ghép nhãn thành chuỗi |
| `evaluation.py` | Tách validation, chọn `k`, confusion matrix, accuracy/macro-F1 và CSV |
| `visualization.py` | Lưu confusion matrix và ví dụ dự đoán |
| `tests/` | Kiểm thử đơn vị và tích hợp trên dữ liệu tổng hợp |
| `data/` | Hướng dẫn nơi đặt dataset, không chứa dữ liệu tải sẵn |
| `results/` | Ghi chú về các file được tạo khi chạy; chưa có kết quả ban đầu |

## Các bước sinh viên cần làm

- [ ] Đọc và tuân thủ điều khoản sử dụng EMNIST trước khi tải dữ liệu.
- [ ] Tải, giải nén và kiểm tra đúng 5 file ByClass theo `data/README.md`.
- [ ] Chạy test phần mềm bằng lệnh trên.
- [ ] Chạy thực nghiệm, giữ lại `metrics.json`, các biểu đồ và CSV thực tế.
- [ ] Thử tối thiểu 5–10 ảnh ký tự riêng; ghi nguồn ảnh, nhãn đúng đã xác nhận,
  dự đoán, trạng thái/nhận xét và nguyên nhân trường hợp sai.
- [ ] Đưa hạn chế subset, ảnh ngoài phân phối và việc không hỗ trợ dấu tiếng Việt
  vào báo cáo.
