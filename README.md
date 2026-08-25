# Dự đoán Giá Nhà Việt Nam — Học máy Truyền thống

## Nội dung

| File | Mô tả |
|---|---|
| `housing_price_prediction.ipynb` | Notebook đầy đủ: 3 biểu đồ phân phối, 5 mô hình học máy, đánh giá bằng 5 độ đo, chọn mô hình tốt nhất. |
| `vietnam_housing_dataset.csv` | Dữ liệu gốc (30,229 bất động sản). |
| `app.py` | Ứng dụng độc lập triển khai mô hình tốt nhất (Random Forest Regressor), dự đoán giá cho 3 trường hợp mẫu. |
| `housing_price_model.joblib` | Mô hình đã huấn luyện + schema đặc trưng (dùng chung cho notebook và app). |
| `requirements.txt` | Thư viện cần thiết. |
| `dist1_price.png`, `dist2_area.png`, `dist3_bedrooms.png` | 3 biểu đồ phân phối. |
| `model_comparison.png`, `best_model_pred_vs_actual.png` | Biểu đồ so sánh mô hình và dự đoán vs thực tế. |

## Tóm tắt kết quả

- **Bài toán:** Hồi quy — dự đoán `Price` (giá nhà, tỷ VNĐ) từ diện tích, mặt tiền, đường vào, số tầng,
  số phòng, hướng nhà, pháp lý, nội thất, khu vực.
- **5 mô hình đã thử nghiệm:** Linear Regression, k-NN Regressor, Decision Tree Regressor,
  Random Forest Regressor, Gradient Boosting Regressor.
- **5 độ đo đánh giá:** MAE, MSE, RMSE, R², MAPE.
- **Mô hình tốt nhất:** Random Forest Regressor (R² cao nhất, MAE/RMSE thấp nhất trong lần chạy này —
  xem bảng so sánh trong notebook, mục 3).

## Cách chạy

### 1. Cài đặt môi trường

```bash
python3 -m pip install -r requirements.txt
```

### 2. Chạy notebook (huấn luyện lại toàn bộ, tạo lại `housing_price_model.joblib`)

```bash
jupyter notebook housing_price_prediction.ipynb
```

Chạy toàn bộ các cell từ trên xuống (`Kernel → Restart & Run All`).

### 3. Chạy ứng dụng độc lập

Sau khi notebook đã chạy ít nhất một lần (để tạo `housing_price_model.joblib`):

```bash
python3 app.py
```

Ứng dụng sẽ nạp mô hình đã lưu và in ra giá dự đoán cho 3 căn nhà mẫu, minh họa pipeline hoàn chỉnh
**Dữ liệu đầu vào → Biểu diễn đặc trưng → Mô hình → Dự đoán giá**.

## Ghi chú tiền xử lý

- Các cột số bị thiếu (`Frontage`, `Access Road`, `Floors`, `Bedrooms`, `Bathrooms`) được điền bằng
  **trung vị**.
- Các cột phân loại bị thiếu được điền nhãn `"Unknown"`.
- Trường `Address` được rút gọn thành `city` (tỉnh/thành phố), gộp các thành phố hiếm thành `"Khac"`.
- Loại bỏ 0.5% giá trị `Price` cực đoan nhất để mô hình ổn định hơn.
- `random_state=42` được dùng xuyên suốt để đảm bảo khả năng tái lập kết quả.
