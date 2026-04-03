---
# Enron Spam Filter (Đồ án 19)

Dự án xây dựng hệ thống phân loại Email (Spam/Ham) sử dụng phương pháp **TF-IDF** kết hợp với các mô hình Machine Learning (**Multinomial Naive Bayes** và **Linear SVM**). Hệ thống hỗ trợ song ngữ Anh - Việt.
---
## 🛠 Tính năng chính

* **Pipeline hoàn chỉnh:** Từ xử lý thô, làm sạch văn bản, trích xuất đặc trưng đến huấn luyện và đánh giá.
* **Hỗ trợ song ngữ:** Xử lý tốt dữ liệu Tiếng Anh và Tiếng Việt nhờ bộ stopword tùy chỉnh.
* **Giao diện trực quan:** Tích hợp Demo qua Streamlit Web App.
* **Tối ưu hóa:** Tự động chọn mô hình tốt nhất dựa trên chỉ số F1-score.

---

## Cấu trúc thư mục (Project Structure)

```text
├── data/
│   ├── bilingual/          # Dữ liệu mẫu (Ham/Spam) Anh-Việt
│   ├── stopwords_vi.txt    # Danh sách stopword Tiếng Việt
│   ├── interim/            # Dữ liệu trung gian sau khi parse
│   └── processed/          # Dữ liệu đã split (train/test)
├── models/                 # Lưu trữ các file .joblib sau khi training
├── reports/                # Kết quả đánh giá và biểu đồ metrics
├── src/                    # Mã nguồn chính (Processing, Training, Predict)
├── app.py                  # Giao diện Web Demo (Streamlit)
└── requirements.txt        # Các thư viện cần thiết

```

---

## Hướng dẫn cài đặt & Chạy dự án

### 1. Thiết lập môi trường

Mở Windows PowerShell và chạy các lệnh sau:

```powershell
# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt môi trường
.venv\Scripts\activate

# Cài đặt thư viện
pip install -r requirements.txt

```

### 2. Luồng thực thi (Workflow)

Để hoàn thành một chu trình dự án, hãy thực hiện theo thứ tự:

1. **Huấn luyện mô hình (Train):**

```powershell
# Sử dụng tập dữ liệu song ngữ mặc định
python -m src.models.train --data-dir data\bilingual

```

2. **Đánh giá (Evaluate):** Kiểm tra độ chính xác trên tập test.

```powershell
python -m src.models.evaluate --model models/best_model.joblib

```

3. **Dự đoán nhanh (Predict):** Kiểm tra thử với 1 nội dung cụ thể.

```powershell
python -m src.models.predict --text "Chúc mừng! Bạn đã trúng thưởng 100 triệu đồng."

```

4. **Giao diện Web (Demo):**

```powershell
streamlit run app.py

```

---

## 🇻🇳 Hỗ trợ Tiếng Việt

Hệ thống sử dụng bộ lọc stopword tại `data/stopwords_vi.txt`.

* Để điều chỉnh danh sách lọc: Sửa trực tiếp file `.txt`.
* Để bật/tắt: Cấu hình biến `USE_VIETNAMESE_STOPWORDS` trong `src/config.py`.

---

## Kết quả đầu ra (Outputs)

Sau khi chạy xong pipeline, các file sau sẽ được khởi tạo/cập nhật:

* `models/best_model.joblib`: Mô hình tối ưu nhất dùng cho Production.
* `reports/results/metrics.csv`: Bảng so sánh chi tiết hiệu năng giữa NB và SVM.
* `data/processed/`: Các tập dữ liệu đã được chuẩn hóa.

---

## Lưu ý

* Nếu bạn thay đổi phiên bản **scikit-learn**, hãy thực hiện huấn luyện lại mô hình để tránh lỗi tương thích.
* Dữ liệu đầu vào cần được đặt đúng cấu trúc thư mục `ham/` và `spam/`.

---
