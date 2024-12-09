
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib

def drop_last_sem(df):
  drl_cols = [col for col in df.columns if col.endswith('_drl') ]
  dtbhk_cols = [col for col in df.columns if col.endswith('_dtbhk') ]
  sotchk_cols = [col for col in df.columns if col.endswith('_sotchk') ]
  for index, row in df.iterrows():
    for idx, val in enumerate(row[dtbhk_cols]): # 1->12
      last_sem = idx
      drl_name = drl_cols[last_sem]
      dtbhk_name = dtbhk_cols[last_sem]
      sotchk_name = sotchk_cols[last_sem]

      if pd.isna(val):
        break

    if pd.isna(row[drl_name]) and pd.isna(row[dtbhk_name]) and pd.isna(row[sotchk_name]):
      sem_to_drop = last_sem-1
      drl_name = drl_cols[sem_to_drop]
      dtbhk_name = dtbhk_cols[sem_to_drop]
      sotchk_name = sotchk_cols[sem_to_drop]

    df.at[index, drl_name] = np.nan
    df.at[index, dtbhk_name] = np.nan
    df.at[index, sotchk_name] = np.nan
  return df

scaler_cluster = joblib.load('resource/scaler_cluster.pkl')
kmeans_cluster = joblib.load('resource/kmeans_perfCluster.pkl')

def normalize_column(df, column_names):
    for col in column_names:
        # Convert column to numeric values, coercing any errors (non-numeric values) to NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')
        

        # Check if the column contains all NaN values (skip normalization if true)
        if df[col].isna().all():
            print(f"Skipping normalization for column '{col}' because it contains all NaN values.")
            continue

        # Min and max for the column, excluding NaN values (use `skipna=True`)
        min_val = df[col].min()
        max_val = df[col].max()

        # Prevent division by zero if min == max (which happens when all non-NaN values are identical)
        if min_val == max_val:
            print(f"Skipping normalization for column '{col}' because all non-NaN values are identical.")
            continue

        # Apply normalization formula, ignoring NaNs in the calculation
        df[col] = (df[col] - min_val) / (max_val - min_val)

    return df

def calculate_perfCluster(df, columns ):
  scaled_data = scaler_cluster.fit_transform(df[columns])

  df['performance_cluster'] = kmeans_cluster.predict(scaled_data)
  return df


def calculate_slope(scores):
    # Filter out NaN values and their corresponding indices
    valid_scores = [(i, score) for i, score in enumerate(scores) if not np.isnan(score)]

    # If there are less than two valid scores, return NaN because you can't calculate a slope
    if len(valid_scores) < 2:
        return np.nan

    # Unzip the valid (non-NaN) scores and their corresponding indices
    x = np.array([i for i, _ in valid_scores]).reshape(-1, 1)  # Indices (semester)
    y = np.array([score for _, score in valid_scores])  # Scores

    # Fit a linear regression model to get the slope
    model = LinearRegression()
    model.fit(x, y)

    return model.coef_[0]  # Return the slope

def feature_engineering(df):
  w_dtbhk = 0.5
  w_sotchk = 0.3
  w_drl = 0.2

  dtbhk_cols = [col for col in df.columns if col.endswith('_dtbhk') ]
  sotchk_cols = [col for col in df.columns if col.endswith('_sotchk') ]
  drl_cols = [col for col in df.columns if col.endswith('_drl') ]

  df['AverageDTBHK'] = df[dtbhk_cols].mean(axis=1, skipna=True)
  df['AverageSOTCHK'] = df[sotchk_cols].mean(axis=1, skipna=True)
  df['AverageDRL'] = df[drl_cols].mean(axis=1, skipna=True)
  df['NumSems'] = df[dtbhk_cols].notna().sum(axis=1)

  #print('*'*16)
  #print('Calculate Average done!')

  segment_cols = ['AverageDTBHK','AverageSOTCHK','AverageDRL', 'NumSems']
  df = calculate_perfCluster(df, segment_cols)

  #print('*'*16)
  #print('Calculate PerfCluster done!')

  df['WeightedPerformance'] = (w_dtbhk*df['AverageDTBHK'] + w_sotchk*df['AverageSOTCHK'] + w_drl*df['AverageDRL'])/(w_dtbhk+w_sotchk+w_drl)

  df['SlopeDTBHK'] = df.apply(lambda row: calculate_slope(row[dtbhk_cols]), axis=1)
  df['SlopeSOTCHK'] = df.apply(lambda row: calculate_slope(row[sotchk_cols]), axis=1)
  df['SlopeDRL'] = df.apply(lambda row: calculate_slope(row[drl_cols]), axis=1)

  #print('*'*16)
  #print('Calculate Performance done!')

  df['StabilityDTBHK'] = df[dtbhk_cols].std(axis=1, skipna=True)
  df['StabilitySOTCHK'] = df[sotchk_cols].std(axis=1, skipna=True)
  df['StabilityDRL'] = df[drl_cols].std(axis=1, skipna=True)

  #print('*'*16)
  #print('Calculate Stability done!')

  df['SlopeToStabilityDTBHK'] = df['SlopeDTBHK'] / df['StabilityDTBHK']
  df['SlopeToStabilitySOTCHK'] = df['SlopeSOTCHK'] / df['StabilitySOTCHK']
  df['SlopeToStabilityDRL'] = df['SlopeDRL'] / df['StabilityDRL']

  #print('*'*16)
  #print('Calculate SlopeToAverage done!')

  return df.drop(dtbhk_cols+sotchk_cols+drl_cols, axis=1)

# Get input from userd
def get_input_from_user():
  cchv_options = ['',
                  'Bị cảnh cáo vì ĐTB học kỳ',
                  'Bị cảnh cáo vì đóng học phí trễ',
                  'Được xem xét hạ mức',
                  'Bị cảnh cáo vì ĐTB và trễ học phí',
                  'Bị cảnh cáo vì đtb 2 học kỳ liên tiếp<4']

  # Define place holder

  drl_df = pd.DataFrame([
      {
      'HK1_drl': 80,
      'HK2_drl': np.nan,
      'HK3_drl': np.nan,
      'HK4_drl': np.nan,
      'HK5_drl': np.nan,
      'HK6_drl': np.nan,
      'HK7_drl': np.nan,
      'HK8_drl': np.nan,
      'HK9_drl': np.nan,
      'HK10_drl': np.nan,
      'HK11_drl': np.nan,
      'HK12_drl': np.nan,
      }
  ])

  dtbhk_df = pd.DataFrame([
      {
      'HK1_dtbhk': 8,
      'HK2_dtbhk': np.nan,
      'HK3_dtbhk': np.nan,
      'HK4_dtbhk': np.nan,
      'HK5_dtbhk': np.nan,
      'HK6_dtbhk': np.nan,
      'HK7_dtbhk': np.nan,
      'HK8_dtbhk': np.nan,
      'HK9_dtbhk': np.nan,
      'HK10_dtbhk': np.nan,
      'HK11_dtbhk': np.nan,
      'HK12_dtbhk': np.nan,
      }
  ])

  sotchk_df = pd.DataFrame([
      {
      'HK1_sotchk': 20,
      'HK2_sotchk': np.nan,
      'HK3_sotchk': np.nan,
      'HK4_sotchk': np.nan,
      'HK5_sotchk': np.nan,
      'HK6_sotchk': np.nan,
      'HK7_sotchk': np.nan,
      'HK8_sotchk': np.nan,
      'HK9_sotchk': np.nan,
      'HK10_sotchk': np.nan,
      'HK11_sotchk': np.nan,
      'HK12_sotchk': np.nan,
      }
  ])

  # Get 'diem ren luyen'
  st.write("## Điểm rèn luyện")
  edited_drl = st.data_editor(drl_df)

  # Get 'diem trung binh hoc ki' 
  st.write("## Điểm trung bình học kì")
  edited_dtbhk = st.data_editor(dtbhk_df)

  # Get so tin chi hoc ki
  st.write("## Số tín chỉ học kì")
  edited_sotchhk = st.data_editor(sotchk_df)

  avsc_col, cchv_col = st.columns(2)

  # Get avsc
  with avsc_col:
      st.write("## Anh Văn Sơ cấp")
      pass_avsc = st.radio(
          "Đã pass Anh văn sơ cấp? 👇",
          ["Chưa rõ", "Đã pass", "Chưa"],
          key="visibility",)
      if pass_avsc == "Chưa rõ":
          pass_avsc = np.nan
      elif pass_avsc == "Đã pass":
          pass_avsc = 'True'
      else:
          pass_avsc = 'False'

  # Get cchv
  with cchv_col:

      st.write("## Cảnh cáo học vụ")
      allow_cchv = st.checkbox('Bị cảnh cáo học vụ')

      if allow_cchv:
          cchv_1_select = st.selectbox("CCHV_1", options=cchv_options, label_visibility='hidden')
          cchv_2_select = st.selectbox("CCHV_2", options=cchv_options, label_visibility='hidden')
          cchv_3_select = st.selectbox("CCHV_3", options=cchv_options, label_visibility='hidden')
          cchv_4_select = st.selectbox("CCHV_4", options=cchv_options, label_visibility='hidden')
          cchv_5_select = st.selectbox("CCHV_5", options=cchv_options, label_visibility='hidden')
          cchv_6_select = st.selectbox("CCHV_6", options=cchv_options, label_visibility='hidden')

          cchv_df = pd.DataFrame([
              {
                  'CCHV_1': cchv_1_select,
                  'CCHV_2': cchv_2_select,
                  'CCHV_3': cchv_3_select,
                  'CCHV_4': cchv_4_select,
                  'CCHV_5': cchv_5_select,
                  'CCHV_6': cchv_6_select,
              }
          ])

      else:
          cchv_df = pd.DataFrame([
              {
                  'CCHV_1': np.nan,
                  'CCHV_2': np.nan,
                  'CCHV_3': np.nan,
                  'CCHV_4': np.nan,
                  'CCHV_5': np.nan,
                  'CCHV_6': np.nan,
              }
          ])

  avsc_df = pd.DataFrame([
      {
          'pass_avsc': pass_avsc,
      }
  ])

  input_dataframe = pd.concat([
                            edited_drl, 
                            edited_dtbhk, 
                            edited_sotchhk, 
                            avsc_df,
                            cchv_df
                            ], axis=1)
  return input_dataframe