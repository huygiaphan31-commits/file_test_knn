# Chẩn đoán nhận diện: sample-02.png

- Kết quả dòng từ pipeline hiện tại: `l c7lojl mtLiL1 I`
- Số vùng phân đoạn: 14
- Chưa có ground truth; không kết luận đúng/sai.
- Cấu hình: prediction-only, 12.400 mẫu train, `k=3` (chọn bằng validation).

| Vùng | Dự đoán | Hộp bao (x0,y0–x1,y1), kích thước crop | Crop chuẩn hóa 28×28 | 5 láng giềng gần nhất (nhãn, khoảng cách Euclid) |
|---:|:---:|---|---|---|
| 1 | `l` | (0, 132)–(7, 147), 9×19 | [crop_01_normalized_28x28.png](crop_01_normalized_28x28.png) | l (3.9237), I (4.1329), l (4.1638), l (4.2061), 1 (4.2358) |
| 2 | `c` | (68, 72)–(93, 111), 35×49 | [crop_02_normalized_28x28.png](crop_02_normalized_28x28.png) | L (4.3733), c (4.4496), c (4.4656), C (4.8704), J (4.9298) |
| 3 | `7` | (83, 70)–(92, 88), 13×22 | [crop_03_normalized_28x28.png](crop_03_normalized_28x28.png) | 7 (2.9875), j (3.1174), I (3.3301), j (3.4559), i (3.4978) |
| 4 | `l` | (102, 52)–(125, 114), 37×76 | [crop_04_normalized_28x28.png](crop_04_normalized_28x28.png) | l (4.5493), L (4.5572), j (4.5837), f (4.5902), T (4.6420) |
| 5 | `o` | (131, 80)–(151, 106), 26×32 | [crop_05_normalized_28x28.png](crop_05_normalized_28x28.png) | o (4.7277), 0 (4.9129), o (4.9199), 0 (4.9545), o (4.9997) |
| 6 | `j` | (148, 86)–(175, 145), 41×73 | [crop_06_normalized_28x28.png](crop_06_normalized_28x28.png) | j (4.2673), j (4.5384), I (4.6482), j (4.6675), j (4.6784) |
| 7 | `l` | (158, 84)–(165, 104), 11×24 | [crop_07_normalized_28x28.png](crop_07_normalized_28x28.png) | l (2.0817), l (2.2558), l (2.4645), l (2.4792), i (2.6148) |
| 8 | `m` | (202, 82)–(210, 86), 12×8 | [crop_08_normalized_28x28.png](crop_08_normalized_28x28.png) | m (4.1277), m (4.2890), N (4.2949), m (4.3306), m (4.3839) |
| 9 | `t` | (204, 93)–(210, 102), 10×13 | [crop_09_normalized_28x28.png](crop_09_normalized_28x28.png) | t (3.5939), Y (3.9191), t (3.9316), P (3.9356), f (3.9441) |
| 10 | `L` | (213, 70)–(229, 107), 24×45 | [crop_10_normalized_28x28.png](crop_10_normalized_28x28.png) | L (3.3395), L (3.6937), b (3.8180), l (3.8516), b (3.9233) |
| 11 | `i` | (234, 42)–(259, 107), 41×81 | [crop_11_normalized_28x28.png](crop_11_normalized_28x28.png) | i (3.7660), i (3.8921), 1 (3.8934), l (3.9131), 1 (4.0941) |
| 12 | `L` | (268, 84)–(279, 106), 17×28 | [crop_12_normalized_28x28.png](crop_12_normalized_28x28.png) | L (3.9108), L (4.0617), L (4.4914), L (4.5692), L (4.6638) |
| 13 | `1` | (281, 82)–(287, 107), 12×31 | [crop_13_normalized_28x28.png](crop_13_normalized_28x28.png) | 1 (2.3828), l (2.5806), i (2.5930), 1 (2.6633), I (2.6699) |
| 14 | `I` | (410, 140)–(412, 146), 4×10 | [crop_14_normalized_28x28.png](crop_14_normalized_28x28.png) | I (3.4013), I (3.6098), i (3.6226), I (3.6711), I (3.7192) |

Ảnh mask, vùng đánh số và crop thô nằm trong thư mục `segmentation/`.
Láng giềng là các mẫu train gần nhất theo khoảng cách Euclid, không phải nhãn đúng của ảnh.
