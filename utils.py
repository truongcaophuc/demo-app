import joblib
import numpy as np
import pandas as pd
import streamlit as st

def compute_skew(row):
    row_without_nan = row.dropna()
    if len(row_without_nan) >= 3:
        return row_without_nan.skew()
    else:
        return 0
    

def compute_kurt(row):
    row_without_nan = row.dropna()
    if len(row_without_nan) >= 3:
        return row_without_nan.kurt()
    else:
        return 3


def cal_std(row):
  #Standard deviration
  return np.nanstd(row)

def cal_diff(row):
    notna = row.notna()  # Xác định vị trí của giá trị not NaN trong hàng
    tmp=row[notna]
    if len(tmp)>=2:
      diff = tmp[-1] - tmp[- 2] 
      return diff
    else: 
      return 0 

def feature_engineering(df):
  dtb_columns = [col for col in df.columns if col.endswith('_dtbhk')]
  df['dtb_mean']= np.nanmean(df[dtb_columns],axis=1)
  df['dtb_std']= df[dtb_columns].apply(cal_std, axis=1)
  df['dtb_diff']=df[dtb_columns].apply(cal_diff, axis=1)
#  df['dtb_skew']= df[dtb_columns].apply(compute_skew, axis= 1)
#  df['dtb_kurt']= df[dtb_columns].apply(compute_kurt, axis= 1)
#  df['ratio_greater7.5']=df[dtb_columns].apply(lambda row: row[row.ge(7.5)].count() / row.dropna().count(), axis=1)
#  df['ratio_greater5']=df[dtb_columns].apply(lambda row: row[row.ge(5)].count() / row.dropna().count(), axis=1)

  drl_columns = [col for col in df.columns if col.endswith('_drl')]

  df['drl_mean']= np.nanmean(df[drl_columns],axis=1)
  df['drl_std']= df[drl_columns].apply(cal_std, axis=1)
#  df['drl_skew']= df[drl_columns].apply(compute_skew, axis= 1)
#  df['drl_kurt']= df[drl_columns].apply(compute_kurt, axis= 1)
#  df['ratio_excellent']=df[drl_columns].apply(lambda row: row[row.ge(90)].count() / row.dropna().count(), axis=1)
#  df['ratio_undergood']=df[drl_columns].apply(lambda row: row[row.le(80)].count() / row.dropna().count(), axis=1)

  sotchk_columns = [col for col in df.columns if col.endswith('_sotchk')]
  df['sotchk_mean']= np.nanmean(df[sotchk_columns],axis=1)

  #df['sotchk_std']= np.nanstd(df[drl_columns],axis=1)
  #df['sotchk_skew']= df[drl_columns].apply(compute_skew, axis= 1)
  #df['sotchk_kurt']= df[drl_columns].apply(compute_skew, axis= 1)
  return df.drop(dtb_columns+drl_columns+sotchk_columns, axis=1)

# Get input from userd
def get_input_from_user():
  cchv_options = ['',
                  'Bị cảnh cáo vì ĐTB học kỳ',
                  'Bị cảnh cáo vì đóng học phí trễ',
                  'Được xem xét hạ mức',
                  'Bị cảnh cáo vì ĐTB và trễ học phí',
                  'Bị cảnh cáo vì đtb 2 học kỳ liên tiếp < 4']

  # Define place holder

  drl_df = pd.DataFrame([
      {
      'HK1_drl': np.nan,
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
      'HK1_dtbhk': np.nan,
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
      'HK1_sotchk': np.nan,
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
          pass_avsc = 1.0
      else:
          pass_avsc = 0.0

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

  input_dataframe = pd.concat([edited_drl, 
                               edited_dtbhk, 
                               edited_sotchhk, 
                               avsc_df,
                               cchv_df,], axis=1)
  return input_dataframe