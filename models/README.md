# Thư mục model AI

Đặt model **Google Teachable Machine** (bản xuất *Tensorflow → Keras*) vào đây:

```
models/
├── keras_model.h5   ← file model
└── labels.txt       ← danh sách lớp (tải cùng lúc với model)
```

Ứng dụng cũng nhận:

- `model.keras` (định dạng Keras 3), hoặc
- thư mục `model.savedmodel/` (bản xuất *Tensorflow → Savedmodel*).

Khi **chưa có** model, ứng dụng tự chạy ở **chế độ DEMO** (so khớp với ảnh mẫu trong `assets/samples/`)
nên mọi chức năng khác vẫn dùng được.

`labels.example.txt` là ví dụ về cách đặt tên lớp. Tên lớp có thể viết có dấu, không dấu hoặc tiếng Anh
(`Búa` / `Bua` / `Hammer`) – ứng dụng tự khớp với kho kiến thức. Nên có thêm lớp `Nền` (ảnh không có dụng cụ).

Sau khi thay model, mở lại ứng dụng để dùng model mới.
