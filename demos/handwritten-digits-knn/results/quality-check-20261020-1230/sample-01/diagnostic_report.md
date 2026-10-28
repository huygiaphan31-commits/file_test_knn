# Chẩn đoán nhận diện: sample-01.png

- Kết quả dòng từ pipeline hiện tại: `kcl il0`
- Số vùng phân đoạn: 6
- Nội dung mong muốn đã được người dùng xác nhận: `Ket qua`.
- Cấu hình: prediction-only, 12.400 mẫu train, `k=3` (chọn bằng validation).

| Vùng | Dự đoán | Hộp bao (x0,y0–x1,y1), kích thước crop | Crop chuẩn hóa 28×28 | 5 láng giềng gần nhất (nhãn, khoảng cách Euclid) |
|---:|:---:|---|---|---|
| 1 | `k` | (154, 111)–(186, 172), 46×75 | [crop_01_normalized_28x28.png](crop_01_normalized_28x28.png) | k (4.0895), k (4.4549), k (4.6057), K (4.6504), K (4.6886) |
| 2 | `c` | (195, 122)–(228, 168), 45×58 | [crop_02_normalized_28x28.png](crop_02_normalized_28x28.png) | c (5.2533), c (5.6546), c (6.2175), c (6.2714), C (6.2860) |
| 3 | `l` | (228, 122)–(247, 168), 31×58 | [crop_03_normalized_28x28.png](crop_03_normalized_28x28.png) | l (3.7135), l (3.9143), l (3.9936), i (4.0188), i (4.0579) |
| 4 | `i` | (298, 139)–(327, 220), 49×101 | [crop_04_normalized_28x28.png](crop_04_normalized_28x28.png) | i (3.0749), I (4.0160), t (4.1381), J (4.1566), 1 (4.1792) |
| 5 | `l` | (328, 142)–(370, 174), 52×42 | [crop_05_normalized_28x28.png](crop_05_normalized_28x28.png) | l (5.1056), I (5.1417), J (5.2392), T (5.2403), Z (5.2556) |
| 6 | `0` | (376, 136)–(403, 170), 35×42 | [crop_06_normalized_28x28.png](crop_06_normalized_28x28.png) | 0 (4.8116), U (5.0521), C (5.0759), 0 (5.1301), 7 (5.1429) |

Ảnh mask, vùng đánh số và crop thô nằm trong thư mục `segmentation/`.
Láng giềng là các mẫu train gần nhất theo khoảng cách Euclid, không phải nhãn đúng của ảnh.
