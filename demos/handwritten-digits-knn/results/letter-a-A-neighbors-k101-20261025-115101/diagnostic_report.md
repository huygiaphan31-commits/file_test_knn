# Chẩn đoán nhầm A/a theo cấu hình web thực tế

## Cấu hình đã chạy

- Dữ liệu: 500 ảnh/lớp, 31.000 mẫu train; 620 mẫu validation.
- Model thực tế chọn `k=101`, validation macro-F1 `0.513365`.
- `app.py` khai báo chế độ `fixed_9`, nhưng `config.py` đặt `FIXED_EXPERIMENT_K = 101`; model nạp từ giao diện dùng hằng số này.
- Chỉ đọc train IDX và mapping. Không đọc test IDX; ảnh mẫu không dùng để train hoặc chọn k.
- Ground truth của các ảnh trong bảng đã được người dùng xác nhận trước đó.

## Kết quả theo ảnh

| Ảnh | Ground truth | Số vùng | Dự đoán từng vùng | Đúng / chấm được |
|---|---|---:|---|---:|
| `test_aAbB.png` | `aAbB` | 4 | `uhhL` | 0/4 |
| `test_abcD.png` | `abcD` | 4 | `cbCD` | 2/4 |
| `eight-characters.png` | `aAbBcCdD` | 9 | `QhbBcC0lD` | Không chấm: 9 vùng so với 8 ký tự |
| `aB3k9C5dA7.png` | `aB3K9C5dA7` | 10 | `jBjk9Cgjhj` | 3/10 |

## Bằng chứng cho A/a

- `test_aAbB.png`, chữ thường `a` → `u`: trong 101 phiếu có `Q:29`, `u:29`, `a:6`; tie-break theo tổng khoảng cách chọn `u`. Top 10 gồm `u, u, q, u, Q, u, U, a, Q, u`.
- `test_aAbB.png`, chữ hoa `A` → `h`: `h:24`, `A:5`; hai láng giềng gần nhất là `A` ở khoảng cách `3.839394` và `A` ở `4.211942`, láng giềng thứ ba là `h` ở `4.323709`.
- `test_abcD.png`, chữ thường `a` → `c`: phiếu `c:27`, `a:22`; các láng giềng đầu tiên gồm ba `a`, nhưng kết quả biểu quyết trên 101 mẫu là `c`.
- `aB3k9C5dA7.png`, chữ thường `a` → `j`: phiếu `j:37`, `J:14`; top 10 không có nhãn `a`.
- Cùng ảnh, chữ hoa `A` → `h`: phiếu `h:35`, `A:5`; hai láng giềng đầu là `A` ở `4.384416` và `4.648429`, láng giềng thứ ba là `h` ở `4.650944`.

Kết quả trên cho thấy nhầm `A→h` có bằng chứng trực tiếp về biểu quyết trên quá nhiều láng giềng: các láng giềng gần nhất là `A`, nhưng nhiều nhãn `h` ở xa hơn lấn át khi lấy 101 phiếu. Với chữ thường `a`, các láng giềng và phiếu khác nhau giữa ảnh; trong một ảnh phiếu nghiêng về `u/Q`, ảnh khác nghiêng về `j/J`, và ảnh `test_abcD` có `a` nằm trong top neighbors nhưng vẫn thua `c` ở tổng phiếu. Đây là biểu hiện phân loại/độ tương đồng dữ liệu, không phải lỗi ánh xạ đã được xác nhận.

## Phân đoạn, tiền xử lý và ánh xạ

- Với `test_aAbB.png`, `test_abcD.png` và `aB3k9C5dA7.png`, số box lần lượt là 4, 4 và 10, khớp số ký tự đã xác nhận; ảnh crop thô vẫn chứa toàn bộ nét chính của A/a. Vì vậy không thấy bằng chứng phân đoạn cắt mất A/a ở các trường hợp này.
- `eight-characters.png` có 9 vùng cho chuỗi 8 ký tự. Không thể ghép chắc chắn vùng với ground truth theo vị trí nên đã loại khỏi phép chấm từng ký tự.
- Ảnh EMNIST được transpose từ thứ tự IDX trước khi chuẩn hóa; glyph EMNIST và crop ảnh ngoài cùng đi qua `normalize_glyph()` để đổi polarity, trừ nền viền, crop nét, resize giữ tỷ lệ trong 20×20, căn giữa canvas 28×28 và scale về `[0,1]`. Crop thô và glyph 28×28 được lưu cạnh báo cáo để kiểm tra trực quan.
- Mapping trong file ByClass khớp: `10→A`, `36→a`, `43→h`, `45→j`. Loader cũng kiểm tra đủ 62 class ID và bộ ký tự duy nhất. Không có bằng chứng về lỗi mapping.
- Chẩn đoán hiện tại chạy k=101; do đó không thể dùng các dự đoán này để kết luận chính xác hành vi khi k=9.

## Kiểm thử và việc cần xử lý

Chạy `python -B -m unittest discover -s tests -t . -p test_app.py`: 10 test, 6 đạt, 1 thất bại, 3 lỗi. Các test hiện có kỳ vọng `k=9`; ba lỗi phát sinh vì validation mock chỉ chứa score đến k=9 nên không có score cho k=101.

Đề xuất tối thiểu, chưa thực hiện: khôi phục `FIXED_EXPERIMENT_K` về `9` để khớp tên chế độ, app và tests; sau đó chạy lại test và lặp chẩn đoán các ảnh A/a bằng model web thật k=9. Không nên sửa phân đoạn/preprocessing dựa trên phép đo k=101 này.

## Artefact

- `per_region_votes_and_neighbors.csv`: box, nhãn thật khi căn chỉnh được, dự đoán, toàn bộ phân bố phiếu và top 10 khoảng cách.
- `neighbors_all_101.csv`: nhãn, khoảng cách và phiếu của toàn bộ 101 láng giềng mỗi crop.
- Các thư mục theo ảnh có ảnh mask, vùng đánh số, crop thô, glyph 28×28; crop A/a và dự đoán h/j có bảng hình top-10 neighbor.

Không sửa mã, cấu hình, preprocessing, phân đoạn, dữ liệu hoặc kết quả lịch sử.
