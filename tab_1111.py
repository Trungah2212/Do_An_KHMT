# import os
# import tkinter as tk
# import pandas as pd
# from tkinter import ttk
# from tkinter import filedialog
# import threading
# import re
# import numpy as np
# import unicodedata
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA

# global_cluster_centers = None   # Biến toàn cục trung tâm cụm
# global_feature_data = None 
# def upload_and_display_file(frame):
#     global original_data_before_scaling
#     frame.df = None  # Khởi tạo dataframe trống 

# # Hàm tải lên và hiển thị file
#     def handle_upload():
#         global df, original_data_before_scaling
#         file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv"), ("Excel Files", "*.xlsx;*.xls"), ("Text Files", "*.txt")])
#         if file_path:
#             # Sử dụng đa luồng để đọc file mà không làm treo giao diện
#             threading.Thread(target=lambda: convert_to_csv_and_display(frame, file_path)).start()
        
#     def convert_to_csv_and_display(frame, file_path):
#         try:
#             if file_path.endswith('.csv'):
#                 df = pd.read_csv(file_path)  # Đọc toàn bộ dữ liệu
#             elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
#                 df = pd.read_excel(file_path)  # Đọc toàn bộ dữ liệu
#             elif file_path.endswith('.txt'):
#                 df = pd.read_csv(file_path, sep='\t', engine='python')  # Đọc toàn bộ dữ liệu
#             else:
#                 raise ValueError("Lỗi file")

#             frame.df = df
#             # Xóa tất cả các dòng hiện có trong treeview
#             tree.delete(*tree.get_children())
#             # Hiển thị dataframe trong treeview
#             tree["column"] = list(df.columns)
#             tree["show"] = "headings"
#             for column in tree["columns"]:
#                 tree.heading(column, text=column, anchor="center")  # Căn giữa tiêu đề cột
#                 tree.column(column, anchor="e")  # Căn dữ liệu trong cột về phải

#             for index, row in df.iterrows():
#                 values = list(row)
#                 for i in range(len(values)):
#                     values[i] = str(values[i]).rjust(10)
#                 tree.insert("", index, values=values)

#             label.config(text="{}".format(file_path))
#         except Exception as e:
#             label.config(text="Đọc file bị lỗi: {}".format(e))

# # Hàm hiển thị thông tin dữ liệu khi nhấn nút
#     def preprocess():
#         if hasattr(frame, 'df') and frame.df is not None:
#             df = frame.df
#             num_rows = len(df)
#             num_empty_cells = df.isnull().sum().sum()
#             num_columns = df.shape[1]
#             column_names = df.columns.tolist()
#             empty_cells_per_column = df.isnull().sum()

#             # Xóa tất cả các dòng hiện có trong treeview để hiển thị thông tin
#             tree.delete(*tree.get_children())
            
#             # Thêm thông tin vào treeview
#             tree["column"] = ["Thông tin", "Giá trị"]
#             tree["show"] = "headings"
#             tree.heading("Thông tin", text="Thông tin", anchor="center")
#             tree.heading("Giá trị", text="Giá trị", anchor="center")

#             # Thêm số dòng
#             tree.insert("", "end", values=("Tổng số dòng", num_rows))
#             # Thêm số ô trống
#             tree.insert("", "end", values=("Tổng số ô trống", num_empty_cells))
#             # Thêm số cột
#             tree.insert("", "end", values=("Tổng số cột", num_columns))
#             # Thêm tên cột
#             tree.insert("", "end", values=("Chi tiết các cột", ", ".join(column_names)))
            
#             # Thêm số ô trống cho mỗi cột
#             for column, empty_cells in empty_cells_per_column.items():
#                 tree.insert("", "end", values=(f"Số ô trống cột {column}", empty_cells))
#         else:
#             tree.delete(*tree.get_children())
#             tree.insert("", "end", values=("Thông báo", "Vui lòng tải lên một tệp CSV trước khi tiền xử lý."))
# # Hàm xử lý dữ liệu
#     def prep():
#         global prep_df
#         if hasattr(frame, 'df') and frame.df is not None:
#             df = frame.df

#         # Làm sạch dữ liệu:
#             # Loại bỏ những dòng trùng lặp
#             def remove_duplicates(df):
#                 df.drop_duplicates(inplace=True)
#                 return df
            
#             # Hàm xóa dòng thiếu quá nhiều dữ liệu
#             def delete_missing_line(df):
#                 quantity = df.isnull().sum(axis=1)
#                 df_cleaned = df[quantity < 3]
#                 return df_cleaned

#             # chuẩn hóa văn bản và sửa lôi chính tả
#             def process_dataframe(df):
#                 processed_df = df
#                 def process_text(text):
#                     text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
#                     text = text.lower()
#                     text = re.sub(r'[^a-z0-9\s]', '', text)
#                     text = re.sub(r'\s', '', text)
#                     return text
#                 for column in processed_df.columns:
#                     processed_df[column] = processed_df[column].apply(lambda x: process_text(x) if isinstance(x, str) else x)
#                 return processed_df
            
#             # Hàm mã hóa giới tính
#             def gender_coding(df):
#                 for column in df.columns:
#                     gender_keywords = ['male', 'female', 'nam', 'nữ', 'm', 'f', 'mister', 'miss', '_', '/']
#                     if df[column].apply(lambda x: str(x).lower() in gender_keywords).any():
#                         mapping_dict = {'male': 0, 'female': 1, 'nam': 0, 'nữ': 1, 'm': 0, 'f': 1, 'mister': 0, 'miss': 1, '_': 0, '/': 1}
#                         df[column] = df[column].map(mapping_dict)
#                 return df
            
#             # Hàm loại bỏ các cột chứa các dữ liệu dài như địa chỉ, họ tên
#             def delete_columns_with_many_text_data(df, threshold=50):
#                 columns_to_drop = [] 
#                 for column in df.columns:  
#                     if pd.api.types.is_string_dtype(df[column]):
#                         num_text_data = df[column].apply(lambda x: isinstance(x, str)).sum()
#                         if num_text_data >= threshold:
#                             columns_to_drop.append(column)
#                 df.drop(columns=columns_to_drop, inplace=True) 
#                 return df
            
#             # Hàm loại bỏ cột dư thừa
#             def drop_id_column(df):
#                 consecutive_increase_threshold = 100 
#                 for col in df.columns:
#                     consecutive_count = 0
#                     prev_value = None
#                     for value in df[col]:
#                         if prev_value is not None and value == prev_value + 1:
#                             consecutive_count += 1
#                             if consecutive_count >= consecutive_increase_threshold:
#                                 df = df.drop(columns=[col])
#                                 break
#                         else:
#                             consecutive_count = 0
#                         prev_value = value
#                 return df
            
            
#             # Tiền xử lý dữ liệu
#             # Hàm xử lý dữ liệu âm
#             def handle_negative_data(df):
#                 for column in df.columns:
#                     if pd.api.types.is_numeric_dtype(df[column]):
#                         df[column] = df[column].apply(lambda x: abs(x) if x < 0 else x)
#                     else:
#                         df[column] = df[column].str.replace(r'^-', '', regex=True)
#                 return df
            
#             # Hàm xử lý dữ liệu giới tính bị thiếu
#             def fill_gender_na(df):
#                 for column in df.columns:
#                     if df[column].isin([0, 1]).sum() >= 1:
#                         gender_column = column
#                         break
#                 else:
#                     return df
#                 p_female = df[gender_column].sum() / len(df[gender_column])
#                 p_male = 1 - p_female
                
#                 # Điền các ô trống trong cột 'Gender' dựa trên phân phối Bernoulli
#                 for index, row in df.iterrows():
#                     if pd.isnull(row[gender_column]):
#                         df.at[index, gender_column] = np.random.choice([0, 1], p=[p_male, p_female])
#                 return df
            
#             # Hàm điền dữ liệu thiếu bằng trung bình cột
#             def fill_missing_values(data):
#                 missing_values = data.isnull()
#                 means = data.mean()
#                 data = data.fillna(means)
#                 return data
            
#             # Làm sạch dữ liệu
#             df = delete_missing_line(df)
#             df = process_dataframe(df)
#             df = gender_coding(df)
#             df = remove_duplicates(df)
#             df = delete_columns_with_many_text_data(df)
#             df = drop_id_column(df)

#             # Tiền xử lý dữ liệu
#             df = handle_negative_data(df)
#             df = fill_gender_na(df)
#             df = fill_missing_values(df)
#             # items_field = df.columns.tolist()  # Gán tên cột sau khi xử lý
#             # combo_box_field['values'] = items_field

#             global prep_df
#             prep_df = df  


#             # Xóa dữ liệu hiện tại trong treeview
#             for item in tree.get_children():
#                 tree.delete(item)
            
#             # Chèn dữ liệu mới
#             tree["column"] = list(prep_df.columns)
#             tree["show"] = "headings"
            
#             # Tạo tiêu đề cột
#             for column in tree["column"]:
#                 tree.heading(column, text=column)
            
#             # Chèn các dòng dữ liệu
#             for index, row in prep_df.iterrows():
#                 tree.insert("", "end", values=list(row))
#         else:
#             pass
#         update_checkboxes(prep_df.columns)
#         select_column_fil()
        
#         def clear_text():
#                 # Đặt trạng thái Treeview thành NORMAL để cho phép chỉnh sửa
#         # tree.config(state=tk.NORMAL)
        
#         # Xóa toàn bộ dữ liệu trong Treeview
#         tree.delete(*tree.get_children())
        
#         # Đặt trạng thái Treeview thành DISABLED để ngăn chặn chỉnh sửa
#         # tree.config(state=tk.DISABLED)
        
        
        
        
