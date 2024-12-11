import warnings
warnings.filterwarnings("ignore")
import config
import numpy as np
import pandas as pd
import streamlit as st
import lightgbm as lgb
import joblib
import pickle
from utils import feature_engineering, get_input_from_user, normalize_column
import os
# Xác định thư mục gốc của dự án (thư mục demo-app)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Đường dẫn tuyệt đối tới các file
model_path = os.path.join(BASE_DIR, "model", "LogisticRegression_best.pkl")
cchv_le_path = os.path.join(BASE_DIR, "resource", "cchv_le.pkl")
avsc_le_path = os.path.join(BASE_DIR, "resource", "avsc_le.pkl")
train_data_path = os.path.join(BASE_DIR, "resource", "train_data.csv")

st.title("Dự đoán tình trạng buộc thôi học của sinh viên 🤔")

# Load the model
model = joblib.load(model_path)

# Load preprocessor 
cchv_le = joblib.load(cchv_le_path)
avsc_le = joblib.load(avsc_le_path)
test_df = pd.read_csv(train_data_path)

temp = get_input_from_user()

predict_btn = st.button("Predict")
if predict_btn:
    placeholder_df = temp
    #placeholder_df = get_input_from_user()
    placeholder_df['label']=2
    # rearrange column
    placeholder_df = placeholder_df[['label'] + [col for col in placeholder_df.columns if col != 'label']]

    input_df = pd.concat([test_df, placeholder_df], ignore_index=True)
    
    input_df = normalize_column(input_df, config.NUM_FEATURES_OG)

    input_df = feature_engineering(input_df) # create new features

    input_df = normalize_column(input_df, config.NUM_FEATURES_NEW)

    input_df[config.CAT_FEATURES] = input_df[config.CAT_FEATURES].fillna('')
    input_df[config.CAT_FEATURES] = input_df[config.CAT_FEATURES].astype(str)
    for cl in config.CAT_FEATURES:
        if cl != 'pass_avsc':
            input_df[cl] = cchv_le.transform(input_df[cl].astype('O'))
    input_df['pass_avsc'] = avsc_le.transform(input_df['pass_avsc'].astype('O'))

    input_data = input_df.tail(1)
    input_data = input_data[config.FEATURES_ORDER]

    print(input_data)

    out = model.predict_proba(input_data)[0]
    print(out)

    pred_idx = np.argmax(out)
    pred_prob = round(out[pred_idx] * 100, 2)

    if pred_idx == 1:
        st.write(f"### Ôi không 😨 {pred_prob}% là khả năng bạn bị buộc thôi học 😰")
        st.snow()

    else:
        st.write(f"### Thật mừng 🤩 vì {pred_prob}% là khả năng bạn ở lại trường đó 😇")
        st.balloons()