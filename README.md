# 📈 Hệ Thống Kiểm Định Chiến Lược Giao Dịch Định Lượng: Kết Hợp EMA & OBV

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Dự án xây dựng ứng dụng Web tương tác trên nền tảng **Streamlit** nhằm kiểm định, phân tích định lượng (Quantitative Backtesting) và trực quan hóa hiệu quả của chiến lược giao dịch kết hợp giữa hai chỉ báo kỹ thuật kinh điển:
- **Đường trung bình động lũy thừa (Exponential Moving Average - EMA)**
- **Khối lượng cân bằng (On-Balance Volume - OBV)**

Được chuẩn hóa dựa trên đề tài nghiên cứu: *"Hoàn thiện Framework Chiến lược EMA & OBV cho Cổ phiếu ACB"* với việc khắc phục triệt để các lỗi học thuật quan trọng: **Hạch toán vị thế (Position Accounting)**, **Phòng chống Look-ahead Bias** và **Đánh giá rủi ro Quá khớp (Overfitting)** giữa mẫu In-Sample (Train) và Out-of-Sample (Test).

---

## 🎯 1. Bối Cảnh Nghiên Cứu & Khắc Phục Góp Ý Của Giảng Viên

Trong phiên bản nghiên cứu trước, hệ thống gặp phải các vấn đề kỹ thuật lớn đã được giảng viên chỉ ra:
1. **Lỗi không hạch toán đúng vị thế giao dịch (Position Accounting):**
   - Thuật toán tối ưu hóa tham số (Hyperopt) chỉ dựa trên Sharpe Ratio mà không kiểm soát số lượng lệnh thực sự được mở.
   - Khi bộ tham số khiến hệ thống gần như không vào lệnh, phương sai lợi nhuận triệt tiêu ($\sigma \to 0$), làm xuất hiện hiện tượng **Sharpe Ratio bùng nổ thành vô cực (`Sharpe = inf`) hoặc `NaN`**. Đây là **hiện tượng giả (artifact)** do thiếu giao dịch, không phản ánh năng lực thực của chiến lược.
   - **Cách giải quyết:** Bổ sung ràng buộc ngặt **`MIN_TRADES >= 5`** vào hàm mục tiêu; nếu số giao dịch không đạt ngưỡng thống kê tối thiểu, hệ thống sẽ loại bỏ bộ tham số đó.
2. **Khối xuất tín hiệu thực chiến không nhận biết trạng thái tài khoản:**
   - Tín hiệu kỹ thuật bản chất là tín hiệu trạng thái (duy trì nhiều phiên). Nếu không theo dõi tài khoản đang nắm giữ cổ phiếu hay tiền mặt, hệ thống có thể liên tục khuyến nghị MUA thêm (dù đã full vị thế) hoặc báo BÁN khống (thị trường Việt Nam không hỗ trợ short selling cổ phiếu cơ sở).
   - **Cách giải quyết:** Theo dõi biến vị thế `current_position` kết hợp với tín hiệu để xuất ra 4 hành động chuẩn xác: **MUA (BUY)**, **BÁN (SELL)**, **TIẾP TỤC NẮM GIỮ (HOLD)**, hoặc **ĐỨNG NGOÀI (STAND ASIDE)**.
3. **Triệt tiêu Look-ahead Bias & Chi phí ma sát:**
   - Áp dụng cơ chế **`shift(1)`** trên toàn bộ tín hiệu để đảm bảo quyết định phiên hôm nay chỉ sử dụng thông tin khi nến phiên hôm trước đã đóng cửa hoàn toàn.
   - Đưa vào mô phỏng đầy đủ: Phí giao dịch (0.2%), Trượt giá (0.1%) và Cắt lỗ cứng Stop-loss (7.0%).

---

## 📊 2. Quy Tắc Vào / Thoát Lệnh Của Chiến Lược

| Chiến Lược | Điều Kiện Mua (Entry Signal) | Điều Kiện Bán (Exit Signal) | Ghi Chú |
| :--- | :--- | :--- | :--- |
| **EMA Riêng Lẻ** | Giá đóng cửa vượt trên đường EMA: <br> `Close > EMA` | Giá đóng cửa rơi xuống dưới EMA: <br> `Close < EMA` | Đi theo xu hướng giá |
| **OBV Riêng Lẻ** | Độ dốc OBV dương qua $N$ phiên: <br> `OBV_Slope > 0` | Độ dốc OBV âm qua $N$ phiên: <br> `OBV_Slope < 0` | Xác nhận áp lực dòng tiền |
| **EMA + OBV Kết Hợp** | **Giá nằm trên EMA VÀ OBV dốc lên:** <br> `(Close > EMA) & (OBV_Slope > 0)` | **Giá cắt xuống dưới EMA:** <br> `Close < EMA` | Lọc bớt tín hiệu nhiễu (Whipsaw) |

*Mọi tín hiệu đều được dịch chuyển 1 phiên: `entries.shift(1)` và `exits.shift(1)`.*

---

## ⚖️ 3. Kết Quả Kiểm Định: Train (2014-2020) vs Test (2021-2023)

Kết quả backtest thực tế trên dữ liệu cổ phiếu ACB cho thấy sự phân hóa sâu sắc:

| Chiến Lược | Tập Dữ Liệu | Bộ Tham Số | Sharpe Ratio | Tổng Lợi Nhuận (%) | Max Drawdown (%) | Số Lệnh Đã Đóng |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **EMA Riêng lẻ** | **Train** (2014-2020) | EMA=36 | **1.0797** | **+211.97%** | -28.69% | 70 |
| **EMA Riêng lẻ** | **Test** (2021-2023) | EMA=36 | **-0.8601** | **-37.69%** | -59.19% | 41 |
| **OBV Riêng lẻ** | **Train** (2014-2020) | OBV=19 | **1.1735** | **+277.80%** | -30.46% | 51 |
| **OBV Riêng lẻ** | **Test** (2021-2023) | OBV=19 | **-0.6135** | **-38.73%** | -62.50% | 38 |
| **EMA + OBV Kết Hợp** | **Train** (2014-2020) | EMA=36, OBV=20 | **1.1514** | **+236.56%** | **-23.87%** *(Thấp nhất)* | 64 |
| **EMA + OBV Kết Hợp** | **Test** (2021-2023) | EMA=36, OBV=20 | **-0.8298** | **-36.47%** | -60.14% | 37 |
| *Benchmark Buy & Hold* | *Test* (2021-2023) | - | - | *+30.34%* | -45.10% | 1 |

> 📌 **Bài học cốt lõi về Quá khớp (Overfitting):**
> Chiến lược hoạt động rất ấn tượng trong giai đoạn 2014–2020 vì thị trường nằm trong **siêu chu kỳ tăng trưởng (Super Bull-trend)**. Tuy nhiên, khi bước sang giai đoạn 2021–2023 (đỉnh 1500 điểm và cú sập lịch sử năm 2022), sự biến động mạnh và các nhịp sideway whipsaw khiến chỉ báo bám xu hướng (EMA/OBV) liên tục bị bẫy mua đỉnh bán đáy. 
> Do đó, nhà đầu tư cần kết hợp thêm **Bộ lọc xu hướng thị trường chung (VN-Index > SMA200 tuần)** trước khi giải ngân.

---

## 🌟 4. Các Tính Năng Nổi Bật Của Web App

1. **Khuyến nghị giao dịch thực chiến thời gian thực:**
   - Tự động nhận diện trạng thái danh mục (đang cầm cổ phiếu hay tiền mặt) để đưa ra khuyến nghị chuẩn xác cho phiên kế tiếp: `MUA MỚI`, `TIẾP TỤC NẮM GIỮ`, `BÁN CHỐT LỜI/CẮT LỖ`, hoặc `ĐỨNG NGOÀI`.
2. **Biểu đồ tương tác cao cấp (Plotly):**
   - Biểu đồ nến/giá, đường EMA, thanh khối lượng OBV Slope và các điểm đánh dấu MUA (tam giác xanh), BÁN (tam giác đỏ).
3. **So sánh đa chiều In-Sample vs Out-of-Sample:**
   - 3 biểu đồ trực quan hóa cột so sánh Sharpe Ratio, Total Return và Max Drawdown.
   - Biểu đồ đường cong tăng trưởng vốn (Equity Curve) song song với Benchmark Buy & Hold.
   - Biểu đồ sụt giảm vốn (Drawdown Underwater chart).
4. **Sổ lệnh chi tiết (Trade Log):**
   - Xem nhật ký từng lệnh: Ngày vào, Ngày ra, Giá vào, Giá ra, Tỷ suất lợi nhuận (%), Thời gian nắm giữ, Lý do thoát lệnh (Tín hiệu hay Dính Stop-loss 7%).
   - Biểu đồ phân phối lợi nhuận (Histogram).
5. **Ma trận quét độ nhạy tham số (Sensitivity Grid Heatmap):**
   - Quét lưới 2 chiều giữa Chu kỳ EMA và Chu kỳ OBV Slope để phát hiện "Vùng cao nguyên bền vững" hay "Điểm nhọn ăn may".
6. **Động cơ mô phỏng kép (Dual-Engine Architecture):**
   - Tích hợp động cơ chuẩn của thư viện `vectorbt`.
   - Trang bị sẵn bộ mô phỏng vector nội bộ (Native Vectorized Simulator) tự động kích hoạt dự phòng nếu môi trường không có vectorbt, đảm bảo ứng dụng **100% không bao giờ bị crash** trên Streamlit Cloud.

---

## 🚀 5. Hướng Dẫn Cài Đặt & Chạy Cục Bộ (Local Machine)

### Bước 1: Clone kho mã nguồn về máy tính
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### Bước 2: Tạo môi trường ảo Python (Khuyến nghị)
```bash
# Tạo virtual environment
python -m venv venv

# Kích hoạt trên Windows:
venv\Scripts\activate

# Hoặc kích hoạt trên macOS/Linux:
source venv/bin/activate
```

### Bước 3: Cài đặt các thư viện cần thiết
```bash
pip install -r requirements.txt
```

### Bước 4: Khởi chạy ứng dụng Streamlit
```bash
streamlit run app.py
```
Trình duyệt sẽ tự động mở giao diện ứng dụng tại địa chỉ: `http://localhost:8501`.

---

## 🌐 6. Hướng Dẫn Deploy Lên Streamlit Community Cloud (Miễn Phí)

Ứng dụng được đóng gói hoàn chỉnh để đưa lên **Streamlit Community Cloud** chỉ với 3 bước:

1. **Đưa mã nguồn lên GitHub:**
   - Đảm bảo repo trên GitHub chứa đủ các file:
     * `app.py` (Mã nguồn chính)
     * `requirements.txt` (Danh sách thư viện phụ thuộc)
     * `ACB.csv` (File dữ liệu mẫu)
     * `README.md` (Tài liệu này)
2. **Đăng nhập vào Streamlit Community Cloud:**
   - Truy cập [share.streamlit.io](https://share.streamlit.io) và đăng nhập bằng tài khoản GitHub của bạn.
3. **Tạo App Mới (New App):**
   - Bấm nút **"Create app"** hoặc **"New app"**.
   - Chọn mục **"I already have an app"**.
   - Điền các thông tin:
     * **Repository:** Chọn kho GitHub bạn vừa đẩy code lên (ví dụ: `username/trading-ema-obv-app`).
     * **Branch:** `main` (hoặc `master`).
     * **Main file path:** `app.py`.
   - Bấm **"Deploy!"**.
4. Chờ hệ thống build trong 1-2 phút. Sau khi hoàn tất, bạn sẽ nhận được một đường link public (ví dụ: `https://your-trading-app.streamlit.app`) để chia sẻ cho giảng viên hoặc cộng đồng!

---

## 📁 7. Cấu Trúc Thư Mục Dự Án

```text
├── app.py                                   # File ứng dụng Web Streamlit chính
├── requirements.txt                         # Các thư viện phụ thuộc (Streamlit, Pandas, Plotly, v.v.)
├── ACB.csv                                  # File dữ liệu lịch sử giá cổ phiếu ACB (2014 - 2023)
├── ACB_Strategy_EMA_OBV_Nhom3_Adjusted.ipynb# Notebook nghiên cứu gốc (Google Colab)
└── README.md                                # Tài liệu hướng dẫn sử dụng và báo cáo học thuật
```

---

## 👨‍💻 8. Tác Giả & Bản Quyền

- **Đơn vị:** Nhóm nghiên cứu Định lượng & Giao dịch thuật toán.
- **Mã nguồn:** Giấy phép MIT License (Sử dụng tự do cho mục đích nghiên cứu và giáo dục).
