import warnings
warnings.filterwarnings("ignore")

import config
import numpy as np
import pandas as pd
import streamlit as st
import lightgbm as lgb
import joblib
import pickle
from utils import feature_engineering, get_input_from_user, drop_last_sem, normalize_column, train_data, new_data


st.title("Dự đoán tình trạng buộc thôi học của sinh viên 🤔")

# Load the model
#model = joblib.load('model/best_model.joblib')
# model = joblib.load('model/lightGBM.joblib')
model = joblib.load('model/RandomForest_best.pkl')

# Load preprocessor 
cchv_le = joblib.load('resource/cchv_le.joblib')
avsc_le = joblib.load('resource/avsc_le.joblib')




input_df = get_input_from_user()

input_df = normalize_column(input_df, config.NUM_FEATURES_OG, train_data)

# input_df = drop_last_sem(input_df)

input_df = feature_engineering(input_df) # create new features

input_df = normalize_column(input_df, config.NUM_FEATURES_NEW, new_data)

input_df[config.CAT_FEATURES] = input_df[config.CAT_FEATURES].fillna('')

input_df[config.CAT_FEATURES] = input_df[config.CAT_FEATURES].astype(str)

for cl in config.CAT_FEATURES:
    if cl != 'pass_avsc':
        input_df[cl] = cchv_le.transform(input_df[cl].astype('O'))
input_df['pass_avsc'] = avsc_le.transform(input_df['pass_avsc'].astype('O'))

#input_df = input_df.dropna()

#input_df = scaler.transform(input_df)

predict_btn = st.button("Predict")
if predict_btn:
    #input_df = input_df.reindex(columns=model.feature_name_)
    # out = model.predict(input_arr)[0]

    print(input_df)

    out = model.predict_proba(input_df)[0]
    print(out)

    pred_idx = np.argmax(out)
    pred_prob = round(out[pred_idx] * 100, 2)

    if pred_idx == 1:
        st.write(f"### Ôi không 😨 {pred_prob}% là khả năng bạn bị buộc thôi học 😰")
        st.snow()

    else:
        st.write(f"### Thật mừng 🤩 vì {pred_prob}% là khả năng bạn ở lại trường đó 😇")
        st.balloons()