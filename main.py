import warnings
warnings.filterwarnings("ignore")

import config
import numpy as np
import pandas as pd
import streamlit as st
import lightgbm as lgb
import joblib
from utils import feature_engineering, get_input_from_user

st.title("Dự đoán tình trạng buộc thôi học của sinh viên 🤔")

# Load the model
model = joblib.load('model/best_model.joblib')
# model = joblib.load('model/lightGBM.joblib')

# Load preprocessor 
cchv_le = joblib.load('resource/cchv_le.joblib')
avsc_le = joblib.load('resource/avsc_le.joblib')
scaler = joblib.load('resource/scaler.joblib')

input_df = get_input_from_user()

input_arr = feature_engineering(input_df)

input_arr[config.CAT_FEATURES] = input_arr[config.CAT_FEATURES].fillna('')

input_arr[config.CAT_FEATURES] = input_arr[config.CAT_FEATURES].astype(str)

for cl in config.CAT_FEATURES:
    if cl != 'pass_avsc':
        input_arr[cl] = cchv_le.transform(input_arr[cl].astype('O'))

input_arr['pass_avsc'] = avsc_le.transform(input_arr['pass_avsc'].astype('O'))

input_arr = scaler.transform(input_arr)

predict_btn = st.button("Predict")
if predict_btn:
    # input_arr = input_arr.reindex(columns=model.feature_name_)
    # out = model.predict(input_arr)[0]
    out = model.predict_proba(input_arr)[0]
    pred_idx = np.argmax(out)
    pred_prob = round(out[pred_idx] * 100, 2)
    if pred_idx == 1:
        st.write(f"### Ôi không 😨 {pred_prob}% là khả năng bạn bị buộc thôi học 😰")
        st.snow()

    else:
        st.write(f"### Thật mừng 🤩 vì {pred_prob}% là khả năng bạn ở lại trường đó 😇")
        st.balloons()