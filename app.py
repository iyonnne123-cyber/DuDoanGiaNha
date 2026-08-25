"""
Ung dung Du doan Gia Nha Viet Nam
====================================
Trien khai mo hinh hoi quy tot nhat (chon tu 5 mo hinh da thu nghiem trong notebook)
thanh mot ung dung doc lap, co the chay ngoai Jupyter Notebook.

Pipeline: Du lieu tho (dict) -> Bieu dien dac trung -> Mo hinh -> Du doan gia nha

Cach chay:
    python app.py
"""

import numpy as np
import pandas as pd
import joblib

MODEL_PATH = "housing_price_model.joblib"


def load_system(model_path: str = MODEL_PATH):
    """Nap mo hinh da huan luyen cung schema dac trung."""
    bundle = joblib.load(model_path)
    return bundle


def build_input_row(raw: dict, bundle: dict) -> pd.DataFrame:
    """
    Chuyen mot ban ghi bat dong san tho (dict) thanh dung bieu dien dac trung
    (one-hot encoding, cac cot dung thu tu) ma mo hinh mong doi.

    Cac truong dau vao co the co (thieu se duoc dien mac dinh):
        Area, Frontage, Access Road, Floors, Bedrooms, Bathrooms (so)
        House direction, Balcony direction, Legal status, Furniture state, city (chuoi)
    """
    row = {c: raw.get(c, np.nan) for c in bundle['num_cols']}
    for c in bundle['num_cols']:
        if pd.isna(row[c]):
            row[c] = 0

    city = raw.get('city', 'Unknown')
    if city not in bundle['top_cities']:
        city = 'Khac'

    cat_raw = {
        'House direction': raw.get('House direction', 'Unknown'),
        'Balcony direction': raw.get('Balcony direction', 'Unknown'),
        'Legal status': raw.get('Legal status', 'Unknown'),
        'Furniture state': raw.get('Furniture state', 'Unknown'),
        'city': city,
    }

    full = {**row, **cat_raw}
    df_row = pd.DataFrame([full])
    df_row = pd.get_dummies(
        df_row,
        columns=['House direction', 'Balcony direction', 'Legal status', 'Furniture state', 'city']
    )
    df_row = df_row.reindex(columns=bundle['feature_columns'], fill_value=0)
    return df_row


def predict_price(bundle: dict, house: dict) -> dict:
    """Du doan gia nha (ty VND) tu mot ban ghi bat dong san tho."""
    row = build_input_row(house, bundle)
    X = bundle['scaler'].transform(row) if bundle['needs_scaling'] else row
    price = bundle['model'].predict(X)[0]
    return {
        "predicted_price_billion_vnd": round(float(price), 3),
        "model_used": bundle['model_name'],
    }


def main():
    bundle = load_system()
    print(f"Da nap mo hinh: {bundle['model_name']}")
    print(f"So dac trung dau vao: {len(bundle['feature_columns'])}\n")

    demo_cases = [
        {
            "Area": 60, "Frontage": 4.5, "Access Road": 6, "Floors": 3,
            "Bedrooms": 3, "Bathrooms": 3, "House direction": "Đông - Nam",
            "Balcony direction": "Đông - Nam", "Legal status": "Have certificate",
            "Furniture state": "Full", "city": "Hà Nội",
        },
        {
            "Area": 100, "Frontage": 8, "Access Road": 12, "Floors": 4,
            "Bedrooms": 5, "Bathrooms": 4, "House direction": "Nam",
            "Balcony direction": "Nam", "Legal status": "Have certificate",
            "Furniture state": "Full", "city": "Hồ Chí Minh",
        },
        {
            "Area": 35, "Frontage": 3, "Access Road": 3, "Floors": 2,
            "Bedrooms": 2, "Bathrooms": 1, "House direction": "Unknown",
            "Balcony direction": "Unknown", "Legal status": "Sale contract",
            "Furniture state": "Basic", "city": "Bình Dương",
        },
    ]

    for i, case in enumerate(demo_cases, 1):
        result = predict_price(bundle, case)
        print(f"Truong hop {i}: {case}")
        print(f"   -> Gia du doan: {result['predicted_price_billion_vnd']} ty VND "
              f"(mo hinh: {result['model_used']})\n")


if __name__ == "__main__":
    main()
