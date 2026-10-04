# 🛠️ AI Xưởng Cơ Khí – Công nghệ 8

Ứng dụng web giáo dục giúp học sinh THCS **nhận dạng dụng cụ cơ khí bằng AI** (qua ảnh hoặc webcam),
tra cứu kiến thức, làm quiz và tích điểm XP để lên cấp.

- 🆓 **Miễn phí hoàn toàn, chạy 100% trên máy** – không dùng LLM, không API trả phí, không phụ thuộc cloud
  (font chữ cũng được đóng gói sẵn, chạy được khi mất mạng).
- 🤖 **Có sẵn model AI đã huấn luyện** (MobileNetV2, giống Teachable Machine) nhận dạng **búa, kìm, tua vít, cờ lê,
  cưa tay, giũa** – tải về là chạy được ngay. Thầy cô có thể tự huấn luyện lại bằng ảnh của lớp mình.
- 🧪 Thiếu model vẫn dùng được: ứng dụng tự chuyển sang *chế độ DEMO*.

---

## ✨ Tính năng

| Trang | Nội dung |
|---|---|
| 🏠 **Trang chủ** | Lời chào, thống kê nhanh, *dụng cụ của ngày*, giới thiệu 3 nhóm dụng cụ |
| 📷 **Nhận dạng** | 4 cách đưa ảnh: **tải ảnh**, **chụp webcam** (trên trình duyệt), **camera trực tiếp** (OpenCV, dự đoán liên tục), **ảnh mẫu**. Kết quả gồm: tên dụng cụ, **độ tin cậy**, nhóm dụng cụ, **công dụng**, **lưu ý an toàn**, **top dự đoán**; cảnh báo khi AI chưa chắc chắn kèm gợi ý chụp lại (OpenCV kiểm tra ảnh tối/mờ); học sinh phản hồi kết quả đúng/sai |
| 📚 **Kho kiến thức** | 15 dụng cụ thuộc 3 nhóm (đo & vạch dấu · tháo lắp & kẹp chặt · gia công): cấu tạo, công dụng, cách dùng, an toàn, “em có biết?”. Tìm kiếm không cần gõ dấu |
| 🧠 **Quiz** | 4 chủ đề (tổng hợp, tên dụng cụ, công dụng, an toàn), 5/10/15 câu. 35 câu soạn sẵn + ~70 câu tự sinh từ kho kiến thức (có câu “nhìn hình đoán tên”). Giải thích sau mỗi câu |
| 🏆 **Tiến độ** | Cấp độ, 11 huy hiệu, lịch sử nhận dạng (kèm ảnh thu nhỏ), biểu đồ điểm quiz, bộ sưu tập dụng cụ, tải/xóa hồ sơ, nạp hồ sơ mẫu |

**Nhiều học sinh dùng chung một máy**: mỗi em có hồ sơ riêng (chọn/tạo ở thanh bên).

### Gamification

| Hoạt động | XP |
|---|---|
| Nhận dạng thành công (AI đủ chắc chắn) | +10 (tối đa 5 lần/dụng cụ/ngày) |
| Khám phá dụng cụ mới bằng AI | +20 thưởng |
| Phản hồi kết quả nhận dạng | +2 |
| Hoàn thành một bài học | +5 (mỗi dụng cụ 1 lần) |
| Câu quiz đúng | +10, từ câu đúng thứ 3 liên tiếp +5 🔥 |
| Bài quiz đúng 100% (≥ 5 câu) | +30 thưởng |

7 cấp độ: Học việc → Thợ tập sự → Thợ phụ → Thợ lành nghề → Thợ cả → Kỹ sư nhí → Bậc thầy cơ khí.

---

## 🚀 Cài đặt & chạy

Yêu cầu: **Python 3.10 – 3.12** (đã kiểm thử với Python 3.12, Streamlit 1.65, TensorFlow 2.20).

### Cách nhanh

```bash
./run.sh          # macOS / Linux
run.bat           # Windows (nháy đúp)
```

Lần đầu script tự tạo môi trường ảo `.venv` và cài thư viện (TensorFlow ~ 500 MB nên mất vài phút).

### Cách thủ công

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Mở trình duyệt tại <http://localhost:8501>. Máy khác trong cùng mạng LAN (vd máy chiếu lớp học)
có thể truy cập qua địa chỉ *Network URL* mà Streamlit in ra.

> 💡 Ứng dụng vẫn chạy được nếu **chưa cài được TensorFlow** – khi đó chỉ có chế độ DEMO.

---

## 🧠 Model AI đi kèm

`models/keras_model.h5` (9,5 MB) + `models/labels.txt` – 7 lớp: Búa, Kìm, Tua vít, Cờ lê, Cưa tay, Giũa, Nền.

**Đánh giá trên 185 ảnh kiểm tra** (ảnh model chưa từng thấy khi học, chạy qua đúng đường xử lý của ứng dụng):

| | Kết quả |
|---|---|
| Độ chính xác tổng | **91,3 %** |
| Khi AI chắc chắn (≥ 60 %, chiếm 79 % số ảnh) | **đúng 97,3 %** – các ảnh còn lại ứng dụng báo “chưa chắc chắn” |
| Búa / Kìm / Tua vít | 91 % / 90 % / 92 % |
| Cờ lê | 80 % |
| Cưa tay / Giũa | 70 % / 86 % (ít ảnh kiểm tra: 10 và 7 ảnh) |
| Nền (không có dụng cụ) | 97 % |

Chi tiết (ma trận nhầm lẫn, số ảnh từng lớp) xem `models/model_info.json`.

**Dữ liệu huấn luyện:** 1.253 ảnh chụp thật có giấy phép tự do – từ Wikimedia Commons, Openverse (Flickr…)
và Open Images V7 – được lọc tự động bằng mô hình CLIP rồi **duyệt tay từng ảnh**. Nguồn, tác giả, giấy phép của
từng ảnh: `dataset/CREDITS.csv`. Ảnh chụp thật trong tab “Ảnh mẫu”: `assets/samples_real/CREDITS.json`.

**Hạn chế cần biết**
- Ảnh huấn luyện là ảnh trên mạng; ảnh webcam trong lớp học (ánh sáng, góc chụp) có thể khác → nếu AI hay nhầm
  với dụng cụ của lớp, hãy **chụp thêm ảnh thật và huấn luyện lại** (bên dưới) – độ chính xác sẽ tăng rõ.
- Cưa tay và giũa có ít ảnh nhất nên kém chắc chắn hơn. Mỏ lết chưa có đủ ảnh nên chưa có trong model
  (mỏ lết thường được nhận là cờ lê).
- Một số ảnh huấn luyện có giấy phép phi thương mại (CC BY-NC) → model dùng cho mục đích **giáo dục, phi thương mại**.

### Tự huấn luyện lại trên máy (không cần Teachable Machine, không cần Internet*)

```bash
# 1. Bỏ ảnh vào dataset/<lớp>/  (bua, kim, tua_vit, co_le, cua_tay, giua, nen – hoặc tạo thư mục lớp mới)
# 2. Huấn luyện (5–15 phút trên máy tính thường), model mới tự lưu vào models/ (model cũ được sao lưu)
python scripts/train_model.py
```

\* Lần đầu cần mạng để tải trọng số MobileNetV2 (~9 MB). Thư mục `dataset/` (bộ ảnh đã lọc) đi kèm dự án; nếu
không có, có thể tạo lại từ đầu:

```bash
pip install -r requirements-dataset.txt               # torch + open_clip (chỉ cho bước lọc dữ liệu)
python scripts/collect_dataset.py --out dataset_raw   # tải ảnh có giấy phép tự do (~6.000 ảnh)
python scripts/clean_dataset.py --raw dataset_raw --out dataset   # lọc bằng CLIP + áp dụng kết quả duyệt tay
python scripts/train_model.py
```

## 🤖 Dùng model Teachable Machine (tùy chọn)

1. Vào <https://teachablemachine.withgoogle.com> → **Image Project** → **Standard image model**.
2. Tạo các lớp: `Búa`, `Kìm`, `Tua vít`, `Cờ lê`, `Cưa tay`, `Giũa` và nên có thêm `Nền` (ảnh không có dụng cụ).
   Tên có dấu, không dấu hay tiếng Anh (`Bua`, `Hammer`) đều được – ứng dụng tự khớp với kho kiến thức.
3. Mỗi lớp 80–200 ảnh, chụp nhiều góc / nền / ánh sáng → **Train Model**.
4. **Export Model** → tab **Tensorflow** → **Keras** → **Download my model** (`converted_keras.zip`).
5. Giải nén, chép `keras_model.h5` và `labels.txt` vào thư mục `models/`, rồi mở lại ứng dụng để dùng model mới.

Ứng dụng cũng nhận `model.keras` hoặc thư mục `model.savedmodel/`. Có thể trỏ tới model ở nơi khác bằng
biến môi trường `XUONG_MODEL_PATH` và `XUONG_LABELS_PATH`.

Muốn AI nhận thêm dụng cụ khác (mỏ lết, ê tô, thước cặp…) chỉ cần thêm lớp (Teachable Machine hoặc thư mục mới trong
`dataset/`) – kho kiến thức đã có sẵn bài học cho 15 dụng cụ.

### Chế độ DEMO (khi chưa có model)

Ứng dụng so khớp ảnh với bộ **ảnh mẫu** (`assets/samples/`) bằng đặc trưng OpenCV (dHash + histogram màu):

- Ảnh mẫu → nhận đúng với độ tin cậy cao, để học sinh trải nghiệm đầy đủ luồng nhận dạng – XP – huy hiệu.
- Ảnh khác → độ tin cậy luôn < 50 % nên ứng dụng báo “chưa chắc chắn” (không đưa ra kết quả sai một cách tự tin).
- Giao diện luôn cho biết khi AI đang dùng ảnh minh họa.

Các trường hợp lỗi đều được xử lý và hiện thông báo dễ hiểu, ứng dụng không bị dừng: thiếu `labels.txt`, file model hỏng,
số lớp của model khác số dòng `labels.txt`, chưa cài TensorFlow, camera không mở được…

> **Lỗi `Unrecognized keyword arguments passed to DepthwiseConv2D: {'groups': 1}`** – lỗi rất hay gặp khi mở file
> Teachable Machine bằng TensorFlow ≥ 2.16 – đã được ứng dụng tự xử lý (làm sạch cấu hình file `.h5` trước khi nạp).

---

## 🗂️ Cấu trúc dự án

```
├── app.py                     # Điểm vào: cấu hình trang, điều hướng, thanh bên
├── core/                      # Logic thuần Python – KHÔNG phụ thuộc Streamlit, có unit test
│   ├── config.py              #   đường dẫn, hằng số, ngưỡng mặc định
│   ├── knowledge.py           #   kho kiến thức + khớp nhãn model ↔ dụng cụ (bỏ dấu tiếng Việt)
│   ├── model.py               #   bộ nhận dạng Teachable Machine / Demo, cài model từ .zip
│   ├── imaging.py             #   Pillow/OpenCV: đọc ảnh (xoay EXIF), tiền xử lý, dHash, đánh giá ảnh
│   ├── quiz.py                #   ngân hàng câu hỏi + sinh câu hỏi tự động + chấm điểm
│   ├── gamification.py        #   luật XP, cấp độ, huy hiệu
│   ├── progress.py            #   thao tác trên hồ sơ: nhận dạng, đọc bài, quiz, phản hồi
│   └── storage.py             #   lưu hồ sơ JSON (ghi nguyên tử, chịu được file hỏng)
├── ui/                        # Lớp giao diện dùng chung
│   ├── state.py               #   session_state, cache tài nguyên, thông báo XP/huy hiệu
│   ├── components.py          #   các khối HTML (thẻ kết quả, thẻ dụng cụ, huy hiệu…)
│   ├── sidebar.py             #   thẻ người chơi, chọn hồ sơ, cài đặt nhận dạng
│   └── dialogs.py             #   hộp thoại bài học chi tiết
├── views/                     # Mỗi file là một trang (hàm render())
│   ├── home.py  recognize.py  library.py  quiz.py  progress_page.py  guide.py
├── data/
│   ├── tools.json             # Kho kiến thức 15 dụng cụ, 3 nhóm
│   ├── quiz_bank.json         # 35 câu hỏi soạn sẵn có giải thích
│   └── sample_profile.json    # Hồ sơ học sinh mẫu (xem trước trang Tiến độ)
├── assets/
│   ├── style.css              # Giao diện tùy biến
│   ├── icons/*.svg            # Hình minh họa 15 dụng cụ
│   ├── samples/*.png          # Ảnh minh họa để thử nhận dạng / chế độ demo
│   └── samples_real/*.jpg     # Ảnh chụp thật (chưa dùng khi huấn luyện) để thử model
├── static/fonts/              # Font Be Vietnam Pro, Baloo 2, Chakra Petch (OFL) – chạy offline
├── models/                    # keras_model.h5 + labels.txt + model_info.json (model đi kèm)
├── dataset/                   # 1.253 ảnh đã lọc dùng để huấn luyện (+ CREDITS.csv)
├── scripts/
│   ├── train_model.py         # Huấn luyện model (MobileNetV2) từ dataset/ -> models/
│   ├── collect_dataset.py     # Tải ảnh huấn luyện có giấy phép tự do
│   ├── clean_dataset.py       # Lọc ảnh bằng CLIP + kết quả duyệt tay (dataset_overrides.json)
│   ├── generate_assets.py     # Vẽ lại icon SVG + ảnh mẫu PNG
│   └── make_sample_profile.py # Tạo lại hồ sơ mẫu
├── tests/                     # pytest: core + model (.h5 giả lập Teachable Machine) + giao diện (AppTest)
├── user_data/profiles/        # Hồ sơ học sinh (tạo tự động khi chạy)
├── requirements.txt
└── run.sh / run.bat
```

**Luồng nhận dạng:** ảnh (upload / webcam / OpenCV) → `imaging.load_image` (xoay EXIF, RGB) →
`Classifier.predict` (cắt giữa 224×224, chuẩn hóa [-1, 1] giống Teachable Machine) → `PredictionResult`
→ `progress.record_scan` (XP, lịch sử, khám phá) → `state.update_profile` (xét huy hiệu, lưu JSON, thông báo) → thẻ kết quả.

---

## 🧩 Tùy biến

- **Thêm/sửa dụng cụ:** sửa `data/tools.json` (thêm `aliases` để khớp với tên lớp trong model).
  Thêm hình minh họa trong `scripts/generate_assets.py` rồi chạy `python scripts/generate_assets.py`
  (dụng cụ chưa có hình sẽ hiện biểu tượng mặc định).
- **Thêm câu hỏi:** thêm vào `data/quiz_bank.json` (`topic`: `ten` / `cong_dung` / `an_toan`, `answer` là chỉ số đáp án đúng).
- **Ngưỡng tin cậy / số dự đoán hiển thị:** thanh bên → ⚙️ *Cài đặt nhận dạng*.
- **Thư mục lưu hồ sơ:** biến môi trường `XUONG_PROFILES_DIR`.

## 🧪 Kiểm thử

```bash
python -m pytest tests -q
```

Bộ test gồm: khớp nhãn tiếng Việt, sinh quiz, luật XP/huy hiệu, lưu trữ (kể cả file hỏng), chế độ demo,
nạp một model `.h5` mô phỏng đúng cấu trúc Teachable Machine bằng **cả tf-keras và Keras 3**, cài model từ `.zip`,
và chạy thử toàn bộ giao diện bằng `streamlit.testing` (nhận dạng ảnh mẫu, làm hết một bài quiz…).

## 🛟 Xử lý sự cố

| Hiện tượng | Cách xử lý |
|---|---|
| Cài TensorFlow lỗi | Kiểm tra Python 3.10–3.12, 64-bit. Ứng dụng vẫn chạy ở chế độ DEMO |
| “Model có N lớp nhưng labels.txt có M nhãn” | Dùng đúng cặp `keras_model.h5` + `labels.txt` tải cùng lúc |
| AI nhận sai tên dụng cụ | Kiểm tra tên lớp trong `labels.txt`; có thể thêm bí danh trong `aliases` ở `tools.json` |
| Webcam trên trình duyệt không hiện | Cho phép quyền camera; trình duyệt chỉ cho dùng camera qua `localhost` hoặc HTTPS |
| Camera trực tiếp không mở | Đóng ứng dụng khác đang dùng camera; macOS: cấp quyền Camera cho Terminal; thử “Số thứ tự camera” 1, 2 |
| Lần đầu mở trang nhận dạng hơi lâu | TensorFlow khởi động lần đầu mất vài giây, các lần sau nhanh |

---

Font chữ: [Be Vietnam Pro](https://github.com/bettergui/BeVietnamPro), [Baloo 2](https://github.com/EkType/Baloo2),
[Chakra Petch](https://github.com/m4rc1e/Chakra-Petch) – giấy phép SIL Open Font License (xem `static/fonts/`).
