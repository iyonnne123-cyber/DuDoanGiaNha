import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = "housing_price_model.joblib"


@st.cache_resource
def load_system(model_path: str = MODEL_PATH):
    """Nạp mô hình và bundle dữ liệu vào bộ nhớ cache."""
    return joblib.load(model_path)


def build_input_row(raw: dict, bundle: dict) -> pd.DataFrame:
    """Chuyển đổi dữ liệu nhập thành đạng one-hot encoding đúng schema."""
    row = {c: raw.get(c, np.nan) for c in bundle["num_cols"]}
    for c in bundle["num_cols"]:
        if pd.isna(row[c]):
            row[c] = 0

    city = raw.get("city", "Unknown")
    top_cities = bundle.get("top_cities", ["Hà Nội", "Hồ Chí Minh", "Bình Dương"])
    if city not in top_cities:
        city = "Khac"

    cat_raw = {
        "House direction": raw.get("House direction", "Unknown"),
        "Balcony direction": raw.get("Balcony direction", "Unknown"),
        "Legal status": raw.get("Legal status", "Unknown"),
        "Furniture state": raw.get("Furniture state", "Unknown"),
        "city": city,
    }

    full = {**row, **cat_raw}
    df_row = pd.DataFrame([full])
    df_row = pd.get_dummies(
        df_row,
        columns=[
            "House direction",
            "Balcony direction",
            "Legal status",
            "Furniture state",
            "city",
        ],
    )
    df_row = df_row.reindex(columns=bundle["feature_columns"], fill_value=0)
    return df_row


def predict_price(bundle: dict, house: dict) -> dict:
    """Dự đoán giá nhà (tỷ VNĐ)."""
    row = build_input_row(house, bundle)
    X = bundle["scaler"].transform(row) if bundle["needs_scaling"] else row
    price = bundle["model"].predict(X)[0]
    return {
        "predicted_price_billion_vnd": round(float(price), 3),
        "model_used": bundle["model_name"],
    }


def main():
    st.set_page_config(
        page_title="Dự đoán Giá Nhà Việt Nam", page_icon="🏠", layout="wide"
    )
    st.title("🏠 Ứng dụng Dự đoán Giá nhà Việt Nam")
    st.write("Nhập thông tin đặc trưng của bất động sản để ước tính giá trị:")

    try:
        bundle = load_system()
    except Exception as e:
        st.error(f"Không thể tải file mô hình {MODEL_PATH}: {e}")
        return

    top_cities = list(
        bundle.get("top_cities", ["Hà Nội", "Hồ Chí Minh", "Bình Dương"])
    )
    if "Khác" not in top_cities:
        top_cities.append("Khác")

    directions = [
        "Unknown",
        "Đông",
        "Tây",
        "Nam",
        "Bắc",
        "Đông - Nam",
        "Đông - Bắc",
        "Tây - Nam",
        "Tây - Bắc",
    ]
    legal_list = [
        "Have certificate",
        "Sale contract",
        "Waiting certificate",
        "Unknown",
    ]
    furniture_list = ["Full", "Basic", "Empty", "Unknown"]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📍 Vị trí & Tình trạng")
        city = st.selectbox("Thành phố", top_cities)
        legal_status = st.selectbox("Tình trạng pháp lý", legal_list)
        furniture_state = st.selectbox("Tình trạng nội thất", furniture_list)

    with col2:
        st.subheader("📐 Diện tích & Quy mô")
        area = st.number_input(
            "Diện tích (m²)", min_value=1.0, value=60.0, step=5.0
        )
        frontage = st.number_input(
            "Mặt tiền (m)", min_value=0.0, value=4.5, step=0.5
        )
        access_road = st.number_input(
            "Đường vào (m)", min_value=0.0, value=6.0, step=0.5
        )
        floors = st.number_input("Số tầng", min_value=1, value=3)

    with col3:
        st.subheader("🛋️ Phòng & Hướng")
        bedrooms = st.number_input("Số phòng ngủ", min_value=0, value=3)
        bathrooms = st.number_input("Số phòng vệ sinh", min_value=0, value=3)
        house_dir = st.selectbox("Hướng nhà", directions)
        balcony_dir = st.selectbox("Hướng ban công", directions)

    if st.button("💰 Dự đoán giá ngay", type="primary", use_container_width=True):
        house = {
            "Area": area,
            "Frontage": frontage,
            "Access Road": access_road,
            "Floors": floors,
            "Bedrooms": bedrooms,
            "Bathrooms": bathrooms,
            "House direction": house_dir,
            "Balcony direction": balcony_dir,
            "Legal status": legal_status,
            "Furniture state": furniture_state,
            "city": city,
        }

        result = predict_price(bundle, house)

        st.divider()
        st.success(
            f"### 🏷️ Giá dự đoán: **{result['predicted_price_billion_vnd']} tỷ VNĐ**"
        )
        st.caption(f"Mô hình thực thi: {result['model_used']}")


if __name__ == "__main__":
    main()
