import os
import tkinter as tk
import pandas as pd
from tkinter import ttk
from tkinter import filedialog
import threading
import re
import numpy as np
import unicodedata
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

global_cluster_centers = None   # Biến toàn cục trung tâm cụm
global_feature_data = None 
def upload_and_display_file(frame):
    global original_data_before_scaling
    frame.df = None  # Khởi tạo dataframe trống 

# Hàm tải lên và hiển thị file
    def handle_upload():
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv"), ("Excel Files", "*.xlsx;*.xls"), ("Text Files", "*.txt")])
        if file_path:
            # Sử dụng đa luồng để đọc file mà không làm treo giao diện
            threading.Thread(target=lambda: convert_to_csv_and_display(frame, file_path)).start()
        
        
    def convert_to_csv_and_display(frame, file_path):
        global df, original_data_before_scaling
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)  # Đọc toàn bộ dữ liệu
            elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                df = pd.read_excel(file_path)  # Đọc toàn bộ dữ liệu
            elif file_path.endswith('.txt'):
                df = pd.read_csv(file_path, sep='\t', engine='python')  # Đọc toàn bộ dữ liệu
            else:
                raise ValueError("Lỗi file")

            frame.df = df
            # Xóa tất cả các dòng hiện có trong treeview
            tree.delete(*tree.get_children())
            # Hiển thị dataframe trong treeview
            tree["column"] = list(df.columns)
            tree["show"] = "headings"
            for column in tree["columns"]:
                tree.heading(column, text=column, anchor="center")  # Căn giữa tiêu đề cột
                tree.column(column, anchor="e")  # Căn dữ liệu trong cột về phải

            for index, row in df.iterrows():
                values = list(row)
                for i in range(len(values)):
                    values[i] = str(values[i]).rjust(10)
                tree.insert("", index, values=values)

            label.config(text="{}".format(file_path))
        except Exception as e:
            label.config(text="Đọc file bị lỗi: {}".format(e))

        original_data_before_scaling = df.copy() 
        
        
        
# Hàm hiển thị thông tin dữ liệu khi nhấn nút
    def preprocess():
        if hasattr(frame, 'df') and frame.df is not None:
            df = frame.df
            num_rows = len(df)
            num_empty_cells = df.isnull().sum().sum()
            num_columns = df.shape[1]
            column_names = df.columns.tolist()
            empty_cells_per_column = df.isnull().sum()

            # Xóa tất cả các dòng hiện có trong treeview để hiển thị thông tin
            tree.delete(*tree.get_children())
            
            # Thêm thông tin vào treeview
            tree["column"] = ["Thông tin", "Giá trị"]
            tree["show"] = "headings"
            tree.heading("Thông tin", text="Thông tin", anchor="center")
            tree.heading("Giá trị", text="Giá trị", anchor="center")

            # Thêm số dòng
            tree.insert("", "end", values=("Tổng số dòng", num_rows))
            # Thêm số ô trống
            tree.insert("", "end", values=("Tổng số ô trống", num_empty_cells))
            # Thêm số cột
            tree.insert("", "end", values=("Tổng số cột", num_columns))
            # Thêm tên cột
            tree.insert("", "end", values=("Chi tiết các cột", ", ".join(column_names)))
            
            # Thêm số ô trống cho mỗi cột
            for column, empty_cells in empty_cells_per_column.items():
                tree.insert("", "end", values=(f"Số ô trống cột {column}", empty_cells))
        else:
            tree.delete(*tree.get_children())
            tree.insert("", "end", values=("Thông báo", "Vui lòng tải lên một tệp CSV trước khi tiền xử lý."))

# Hàm xử lý dữ liệu
    def prep():
        global prep_df
        if hasattr(frame, 'df') and frame.df is not None:
            df = frame.df

        # Làm sạch dữ liệu:
            # Loại bỏ những dòng trùng lặp
            def remove_duplicates(df):
                df.drop_duplicates(inplace=True)
                return df
            
            # Hàm xóa dòng thiếu quá nhiều dữ liệu
            def delete_missing_line(df):
                quantity = df.isnull().sum(axis=1)
                df_cleaned = df[quantity < 3]
                return df_cleaned

            # chuẩn hóa văn bản và sửa lôi chính tả
            def process_dataframe(df):
                processed_df = df
                def process_text(text):
                    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
                    text = text.lower()
                    text = re.sub(r'[^a-z0-9\s]', '', text)
                    text = re.sub(r'\s', '', text)
                    return text
                for column in processed_df.columns:
                    processed_df[column] = processed_df[column].apply(lambda x: process_text(x) if isinstance(x, str) else x)
                return processed_df
            
            # Hàm mã hóa giới tính
            def gender_coding(df):
                for column in df.columns:
                    gender_keywords = ['male', 'female', 'nam', 'nữ', 'm', 'f', 'mister', 'miss', '_', '/']
                    if df[column].apply(lambda x: str(x).lower() in gender_keywords).any():
                        mapping_dict = {'male': 0, 'female': 1, 'nam': 0, 'nữ': 1, 'm': 0, 'f': 1, 'mister': 0, 'miss': 1, '_': 0, '/': 1}
                        df[column] = df[column].map(mapping_dict)
                return df
            
            # Hàm loại bỏ các cột chứa các dữ liệu dài như địa chỉ, họ tên
            def delete_columns_with_many_text_data(df, threshold=50):
                columns_to_drop = [] 
                for column in df.columns:  
                    if pd.api.types.is_string_dtype(df[column]):
                        num_text_data = df[column].apply(lambda x: isinstance(x, str)).sum()
                        if num_text_data >= threshold:
                            columns_to_drop.append(column)
                df.drop(columns=columns_to_drop, inplace=True) 
                return df
            
            # Hàm loại bỏ cột dư thừa
            def drop_id_column(df):
                consecutive_increase_threshold = 100 
                for col in df.columns:
                    consecutive_count = 0
                    prev_value = None
                    for value in df[col]:
                        if prev_value is not None and value == prev_value + 1:
                            consecutive_count += 1
                            if consecutive_count >= consecutive_increase_threshold:
                                df = df.drop(columns=[col])
                                break
                        else:
                            consecutive_count = 0
                        prev_value = value
                return df
            
            
            # Tiền xử lý dữ liệu
            # Hàm xử lý dữ liệu âm
            def handle_negative_data(df):
                for column in df.columns:
                    if pd.api.types.is_numeric_dtype(df[column]):
                        df[column] = df[column].apply(lambda x: abs(x) if x < 0 else x)
                    else:
                        df[column] = df[column].str.replace(r'^-', '', regex=True)
                return df
            
            # Hàm xử lý dữ liệu giới tính bị thiếu
            def fill_gender_na(df):
                for column in df.columns:
                    if df[column].isin([0, 1]).sum() >= 1:
                        gender_column = column
                        break
                else:
                    return df
                p_female = df[gender_column].sum() / len(df[gender_column])
                p_male = 1 - p_female
                
                # Điền các ô trống trong cột 'Gender' dựa trên phân phối Bernoulli
                for index, row in df.iterrows():
                    if pd.isnull(row[gender_column]):
                        df.at[index, gender_column] = np.random.choice([0, 1], p=[p_male, p_female])
                return df
            
            # Hàm điền dữ liệu thiếu bằng trung bình cột
            def fill_missing_values(data):
                missing_values = data.isnull()
                means = data.mean()
                data = data.fillna(means)
                return data
            
            # Làm sạch dữ liệu
            df = delete_missing_line(df)
            df = process_dataframe(df)
            df = gender_coding(df)
            df = remove_duplicates(df)
            df = delete_columns_with_many_text_data(df)
            df = drop_id_column(df)

            # Tiền xử lý dữ liệu
            df = handle_negative_data(df)
            df = fill_gender_na(df)
            df = fill_missing_values(df)
            # items_field = df.columns.tolist()  # Gán tên cột sau khi xử lý
            # combo_box_field['values'] = items_field

            global prep_df
            prep_df = df  
            


            # Xóa dữ liệu hiện tại trong treeview
            for item in tree.get_children():
                tree.delete(item)
            
            # Chèn dữ liệu mới
            tree["column"] = list(prep_df.columns)
            tree["show"] = "headings"
            
            # Tạo tiêu đề cột
            for column in tree["column"]:
                tree.heading(column, text=column)
            
            # Chèn các dòng dữ liệu
            for index, row in prep_df.iterrows():
                tree.insert("", "end", values=list(row))
        else:
            pass
        update_checkboxes(prep_df.columns)
        select_column_fil()
        
    
    def clear_text():
        # Đặt trạng thái Treeview thành NORMAL để cho phép chỉnh sửa
        # tree.config(state=tk.NORMAL)
        
        # Xóa toàn bộ dữ liệu trong Treeview
        tree.delete(*tree.get_children())
        
        # Đặt trạng thái Treeview thành DISABLED để ngăn chặn chỉnh sửa
        # tree.config(state=tk.DISABLED)

#Hàm cập nhật checkbox
    def update_checkboxes(columns):
        for widget in frame_checkbox.winfo_children():
            widget.destroy()
        checkbox_vars = []
        for column in columns:
            var = tk.BooleanVar()
            checkbox = ttk.Checkbutton(frame_checkbox, text=column, variable=var)
            checkbox.pack(anchor='w')
            checkbox_vars.append(var)
        # Lưu danh sách các biến kiểm soát checkbox vào một biến global để sử dụng sau này
        global checkbox_vars_global
        checkbox_vars_global = checkbox_vars

# Hàm chọn cột
    def select_column_fil():
        selected_columns = []
        for i, var in enumerate(checkbox_vars_global):
            if var.get():
                selected_columns.append(prep_df.columns[i])
        return selected_columns

# Hàm nhập số cụm bằng tay
    def enter_number_cluster():
        try:
            content = text_clus.get("1.0", "end-1c")  # Lấy nội dung của widget Text
            integer_value = int(content)  # Chuyển đổi nội dung thành số nguyên
            return integer_value
        except ValueError:
            return None
        

# Hàm chọn k tối ưu bằng elbow
    def Elbow(max_k=10):
        try:
            distortions = []
            K = range(1, max_k + 1)
            threshold = 0.1  # Ngưỡng độ giảm distortion không đáng kể
            optimal_k = 1
            for k in K:
                kmeanModel = KMeans(n_clusters=k)
                kmeanModel.fit(prep_df)
                distortions.append(kmeanModel.inertia_)
                if len(distortions) > 1:
                    decrease_ratio = (distortions[-2] - distortions[-1]) / distortions[-2]
                    if decrease_ratio < threshold:
                        optimal_k = k - 1
                        break
            best_k = optimal_k
            text_clus.config(state=tk.NORMAL)
            text_clus.delete(1.0, tk.END)  
            text_clus.insert(tk.END, f"{best_k}")
            return int(text_clus.get("1.0", "end-1c"))
        except ValueError:
            text_clus.config(state=tk.NORMAL)
            text_clus.delete(1.0, tk.END) 
            text_clus.insert(tk.END, "Invalid input")
            text_clus.config(state=tk.DISABLED)
            
# Hàm kmean
    def kmeans_with_selected_columns(prep_df, columns, num_clusters):
        global global_cluster_centers
        scaler = MinMaxScaler()
        prep_df[columns] = scaler.fit_transform(prep_df[columns])

        selected_data = prep_df[columns]
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        clusters = kmeans.fit_predict(selected_data)

        # Lấy tâm của các cụm
        cluster_centers = kmeans.cluster_centers_
        global_cluster_centers = cluster_centers

        data_with_clusters = selected_data.copy()
        data_with_clusters['Cluster'] = clusters
        return data_with_clusters, selected_data
    


# Hàm chọn trường dữ liệu
    def select_data_field(event=None):
        global selected_data_item
        selected_data_item = combo_box_field.get()
        compute_statistics()

# Hàm chọn cụm dữ liệu
    def select_clus_data(event=None):
        global selected_clus_item
        selected_clus_item = combo_box_cluster.get()
        compute_statistics()
   
# Hàm tính toán và hiển thị kmean
    def cluster_data():
        global num_clusters
        num_clusters = enter_number_cluster()
        if num_clusters is not None:
            global selected_columns
            selected_columns = select_column_fil()
            if selected_columns and 'prep_df' in globals():
                global clustered_data, items_field, items_cluster
                clustered_data_tuple = kmeans_with_selected_columns(prep_df, selected_columns, num_clusters)
                clustered_data = clustered_data_tuple[0]  # Accessing the first element of the tuple
                selected_data = clustered_data_tuple[1]  # Accessing the second element of the tuple
                items_field = selected_columns
                items_cluster = list(range(0, num_clusters))
                combo_box_field['values'] = items_field
                combo_box_cluster['values'] = items_cluster
                
                # Hiển thị kết quả phân cụm trong Treeview
                # Xóa dữ liệu hiện tại trong treeview
                for item in tree.get_children():
                    tree.delete(item)
                
                # Tạo lại cột cho Treeview từ dataframe
                tree["column"] = list(clustered_data.columns)
                tree["show"] = "headings"
                
                # Tạo tiêu đề cột
                for column in tree["columns"]:
                    tree.heading(column, text=column)
                
                # Chèn các dòng dữ liệu
                for index, row in clustered_data.iterrows():
                    tree.insert("", "end", values=list(row))
                
                # Hiển thị dữ liệu trong text widget
                tree.config(state=tk.NORMAL)
                tree.delete('1.0', tk.END)
                clustered_data_str = clustered_data.to_string(index=False)
                for line in clustered_data_str.split('\n'):
                    tree.insert(tk.END, line + '\n')
                tree.config(state=tk.DISABLED)
            else:
                tree.config(state=tk.NORMAL)
                tree.delete('1.0', tk.END)
                tree.insert(tk.END, "Vui lòng chọn ít nhất một cột!")
                tree.config(state=tk.DISABLED)
        else:
            tree.config(state=tk.NORMAL)
            tree.delete('1.0', tk.END)
            tree.insert(tk.END, "Số lượng cụm không hợp lệ!")
            tree.config(state=tk.DISABLED)


# Hàm tính toán các chỉ số
    def compute_statistics():
        global global_feature_data, selected_data_item, selected_clus_item
        if 'clustered_data' in globals() and 'prep_df' in globals():
            selected_feature = selected_data_item
            selected_cluster = selected_clus_item
            if selected_cluster is not None and selected_feature:
                display_calcu_result.config(state=tk.NORMAL)
                display_calcu_result.delete('1.0', tk.END)

                # Lấy dữ liệu của cụm và trường đã chọn từ dữ liệu chuẩn hóa
                cluster_data = clustered_data[clustered_data['Cluster'] == int(selected_cluster)]
                feature_data = cluster_data[selected_feature]
                global_feature_data = feature_data

                # Tính toán các giá trị thống kê cho dữ liệu chuẩn hóa
                mean_value = np.mean(feature_data)
                median_value = np.median(feature_data)
                min_value = np.min(feature_data)
                max_value = np.max(feature_data)
                midrange_value = (min_value + max_value) / 2
                std_dev = np.std(feature_data)
                num_values = len(feature_data)
                mode_value = np.argmax(np.bincount(feature_data))
                variance_value = np.var(feature_data)
                quartiles = np.percentile(feature_data, [25, 50, 75])
                q1, q2, q3 = quartiles
                iqr = q3 - q1

                # Lấy lại vị trí của các mẫu thuộc cụm đã chọn từ dữ liệu trước chuẩn hóa
                original_indices = cluster_data.index
                original_feature_data = original_data_before_scaling.loc[original_indices, selected_feature]

                # Tính toán các giá trị thống kê cho dữ liệu trước chuẩn hóa
                mean_value_original = np.mean(original_feature_data)
                median_value_original = np.median(original_feature_data)
                min_value_original = np.min(original_feature_data)
                max_value_original = np.max(original_feature_data)
                midrange_value_original = (min_value_original + max_value_original) / 2
                std_dev_original = np.std(original_feature_data)
                num_values_original = len(original_feature_data)
                # mode_value_original = np.argmax(np.bincount(original_feature_data))
                variance_value_original = np.var(original_feature_data)
                quartiles_original = np.percentile(original_feature_data, [25, 50, 75])
                q1_original, q2_original, q3_original = quartiles_original
                iqr_original = q3_original - q1_original

                # Hiển thị kết quả
                result = f"Kết quả tính dựa theo chọn trường, chọn cụm: \n"
                result += f"\n - Mean (Giá trị trung bình): {mean_value} (Chuẩn hóa), {mean_value_original} (Gốc)"
                result += f"\n - Median (Giá trị trung vị): {median_value} (Chuẩn hóa), {median_value_original} (Gốc)"
                result += f"\n - Midrange (Giá trị trung tâm): {midrange_value} (Chuẩn hóa), {midrange_value_original} (Gốc)"
                result += f"\n - Standard Deviation (Biến động của dữ liệu): {std_dev} (Chuẩn hóa), {std_dev_original} (Gốc)"
                result += f"\n - Number of Values (Danh sách các phần tử): {num_values} (Chuẩn hóa), {num_values_original} (Gốc)"
                # result += f"\n - Mode (Giá trị xuất hiện nhiều nhất): {mode_value} (Chuẩn hóa), {mode_value_original} (Gốc)"
                result += f"\n - Variance (Phương sai): {variance_value} (Chuẩn hóa), {variance_value_original} (Gốc)"
                result += f"\n - Q1 (Phân vị thứ 25): {q1} (Chuẩn hóa), {q1_original} (Gốc)"
                result += f"\n - Q2 (Phân vị thứ 50): {q2} (Chuẩn hóa), {q2_original} (Gốc)"
                result += f"\n - Q3 (Phân vị thứ 75): {q3} (Chuẩn hóa), {q3_original} (Gốc)"
                result += f"\n - IQR (Dải giữa hai phân vị): {iqr} (Chuẩn hóa), {iqr_original} (Gốc)"
                display_calcu_result.insert(tk.END, result)
                display_calcu_result.config(state=tk.DISABLED)
                return mean_value, median_value, midrange_value, std_dev, num_values, mode_value, variance_value, iqr
            else:
                display_calcu_result.config(state=tk.NORMAL)
                display_calcu_result.delete('1.0', tk.END)
                display_calcu_result.insert(tk.END, "Vui lòng chọn trường và cụm!")
                display_calcu_result.config(state=tk.DISABLED)
                return None
        else:
            display_calcu_result.config(state=tk.NORMAL)
            display_calcu_result.delete('1.0', tk.END)
            display_calcu_result.insert(tk.END, "Không có dữ liệu phân cụm để tính toán!")
            display_calcu_result.config(state=tk.DISABLED)
            return None



# Hàm xuất    
    def export_data():
        global clustered_data, data_original, num_clusters
        if 'clustered_data' in globals() and isinstance(clustered_data, pd.DataFrame):
            try:
                if 'data_original' in globals() and isinstance(data_original, pd.DataFrame):
                    for cluster_num in range(num_clusters):
                        cluster_indices = clustered_data[clustered_data['Cluster'] == cluster_num].index
                        cluster_data = data_original.iloc[cluster_indices]
                        cluster_data['Cluster'] = cluster_num  # Thêm cột Cluster vào dữ liệu gốc

                        filename = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                                                filetypes=[("Excel files", "*.xlsx"),
                                                                        ("CSV files", "*.csv"),
                                                                        ("Text files", "*.txt")],
                                                                title=f"Lưu cụm {cluster_num}")
                        if filename:
                            if filename.endswith(".xlsx"):
                                cluster_data.to_excel(filename, index=False)
                            elif filename.endswith(".csv"):
                                cluster_data.to_csv(filename, index=False)
                            elif filename.endswith(".txt"):
                                cluster_data.to_csv(filename, index=False, sep='\t')
                            else:
                                pass
                        else:
                            pass
            except Exception as e:
                print("Error:", e)
        else:
            pass

# Hàm vẽ biểu đồ kmean
    def plot_clustered_data():
        for widget in frame_chart_kmean.winfo_children():
            widget.destroy()

        global clustered_data, num_clusters, selected_columns, global_cluster_centers  
        if 'clustered_data' in globals() and global_cluster_centers is not None:
            num_samples, num_features = clustered_data[selected_columns].shape
            
            if num_samples < 2 or num_features < 2:
                fig, ax = plt.subplots(figsize=(8, 6))
                colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
                cluster_labels = ['Cụm {}'.format(i) for i in range(num_clusters)]

                for i in range(num_clusters):
                    cluster_points = clustered_data[clustered_data['Cluster'] == i][selected_columns[0]]
                    ax.scatter(cluster_points, np.zeros_like(cluster_points), c=colors[i], marker='s', s=3, label=cluster_labels[i])

                ax.set_xlabel(selected_columns[0])
                ax.get_yaxis().set_visible(False)
                ax.legend()
                plt.grid(linestyle='--', alpha=0.5)
            else:
                pca = PCA(n_components=2)
                clustered_data_2d = pca.fit_transform(clustered_data[selected_columns])
                fig = plt.figure(figsize=(8, 6))
                colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
                cluster_labels = ['Cụm {}'.format(i) for i in range(num_clusters)]
                
                for i in range(num_clusters):
                    cluster_points = clustered_data_2d[clustered_data['Cluster'] == i]
                    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[i], marker='s', s=3, label=cluster_labels[i])
                
                for i in range(num_clusters):
                    cluster_center = np.mean(clustered_data_2d[clustered_data['Cluster'] == i], axis=0)
                    plt.scatter(cluster_center[0], cluster_center[1], c=colors[i], marker='^', s=50, label=f'Trung tâm cụm {i}')
            
                plt.grid(linestyle='--', alpha=0.5)


            canvas = FigureCanvasTkAgg(fig, master=frame_chart_kmean)
            canvas.draw()
            canvas.get_tk_widget().pack(side="top", fill='both')
        else:
            print("Không có dữ liệu phân cụm hoặc tâm cụm.")


# Hàm hiển thị biểu đồ ở cửa sổ mới
    def show_plot_clustered_data():
        global clustered_data, num_clusters, selected_columns, global_cluster_centers  
        if 'clustered_data' in globals() and global_cluster_centers is not None:
            plot_window = tk.Toplevel()
            plot_window.title("Biểu đồ Kmean")
            
            tab_control = ttk.Notebook(plot_window)
            
            # Tab for 2D plot
            tab_2d = ttk.Frame(tab_control)
            tab_control.add(tab_2d, text='2D Plot')
            
            # Tab for multi-dimensional plot
            tab_multi_dim = ttk.Frame(tab_control)
            tab_control.add(tab_multi_dim, text='Multi-dimensional Plot')
            
            tab_control.pack(expand=1, fill='both')
            
            num_samples, num_features = clustered_data[selected_columns].shape
            
            if num_samples < 2 or num_features < 2:
                fig, ax = plt.subplots(figsize=(8, 6))
                colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
                cluster_labels = ['Cụm {}'.format(i) for i in range(num_clusters)]

                for i in range(num_clusters):
                    cluster_points = clustered_data[clustered_data['Cluster'] == i][selected_columns[0]]
                    ax.scatter(cluster_points, np.zeros_like(cluster_points), c=colors[i], marker='s', s=3, label=cluster_labels[i])

                ax.set_xlabel(selected_columns[0])
                ax.get_yaxis().set_visible(False)
                ax.legend()
                plt.grid(linestyle='--', alpha=0.5)
            else:
                # 2D Plot
                pca = PCA(n_components=2)
                clustered_data_2d = pca.fit_transform(clustered_data[selected_columns])
                fig_2d = plt.figure(figsize=(8, 6))
                colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
                cluster_labels = ['Cụm {}'.format(i) for i in range(num_clusters)]
                
                for i in range(num_clusters):
                    cluster_points = clustered_data_2d[clustered_data['Cluster'] == i]
                    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[i], marker='s', s=3, label=cluster_labels[i])
                
                for i in range(num_clusters):
                    cluster_center = np.mean(clustered_data_2d[clustered_data['Cluster'] == i], axis=0)
                    plt.scatter(cluster_center[0], cluster_center[1], c=colors[i], marker='^', s=50, label=f'Trung tâm cụm {i}')
            
                plt.grid(linestyle='--', alpha=0.5)
                plt.legend()

                canvas_2d = FigureCanvasTkAgg(fig_2d, master=tab_2d)
                canvas_2d.draw()
                canvas_2d.get_tk_widget().pack(side="top", fill='both')

                # Multi-dimensional Plot
                fig_multi_dim = plt.figure(figsize=(8, 6))
                
                if num_features >= 3:
                    ax_multi_dim = fig_multi_dim.add_subplot(111, projection='3d')
                    pca_3d = PCA(n_components=3)
                    clustered_data_3d = pca_3d.fit_transform(clustered_data[selected_columns])
                    
                    for i in range(num_clusters):
                        cluster_points = clustered_data_3d[clustered_data['Cluster'] == i]
                        ax_multi_dim.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], c=colors[i], marker='s', s=3, label=cluster_labels[i])
                    
                    for i in range(num_clusters):
                        cluster_center = np.mean(clustered_data_3d[clustered_data['Cluster'] == i], axis=0)
                        ax_multi_dim.scatter(cluster_center[0], cluster_center[1], cluster_center[2], c=colors[i], marker='^', s=50, label=f'Trung tâm cụm {i}')
                    
                    ax_multi_dim.set_xlabel(selected_columns[0])
                    ax_multi_dim.set_ylabel(selected_columns[1])
                    ax_multi_dim.set_zlabel(selected_columns[2])
                    ax_multi_dim.legend()
                    plt.grid(linestyle='--', alpha=0.5)
                else:
                    ax_multi_dim = fig_multi_dim.add_subplot(111)
                    pca_2d = PCA(n_components=2)
                    clustered_data_2d = pca_2d.fit_transform(clustered_data[selected_columns])
                    
                    for i in range(num_clusters):
                        cluster_points = clustered_data_2d[clustered_data['Cluster'] == i]
                        ax_multi_dim.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[i], marker='s', s=3, label=cluster_labels[i])
                    
                    for i in range(num_clusters):
                        cluster_center = np.mean(clustered_data_2d[clustered_data['Cluster'] == i], axis=0)
                        ax_multi_dim.scatter(cluster_center[0], cluster_center[1], c=colors[i], marker='^', s=50, label=f'Trung tâm cụm {i}')
                    
                    ax_multi_dim.set_xlabel(selected_columns[0])
                    ax_multi_dim.set_ylabel(selected_columns[1])
                    ax_multi_dim.legend()
                    plt.grid(linestyle='--', alpha=0.5)
                
                canvas_multi_dim = FigureCanvasTkAgg(fig_multi_dim, master=tab_multi_dim)
                canvas_multi_dim.draw()
                canvas_multi_dim.get_tk_widget().pack(side="top", fill='both')
            
        else:
            print("Không có dữ liệu phân cụm hoặc tâm cụm.")
# Biểu đồ thống kê giá trị
    def show_plot_statistics_value():
        global global_feature_data
        if global_feature_data is not None:

            plot_window = tk.Toplevel()
            plot_window.title("Biểu đồ Kmean")

            feature_data = global_feature_data
            # Create a figure and axis
            fig, ax = plt.subplots()
            # Plot data
            ax.bar(range(len(feature_data)), feature_data)
            # Add title and labels
            ax.set_title(f'Biểu đồ về {selected_data_item} ở cụm {selected_clus_item}')
            ax.set_xlabel('Số liệu')
            ax.set_ylabel('Gía trị')
            # Display the plot in the tkinter window
            canvas = FigureCanvasTkAgg(fig, master=plot_window)  
            canvas.draw()
            canvas.get_tk_widget().pack(side='top', fill='both', expand=True)
        else:
            print("No data to plot. Please select a field and cluster first.")

    def plot_statistics_value():
        global global_feature_data
        if global_feature_data is not None:
            feature_data = global_feature_data
            fig, ax = plt.subplots()
            ax.bar(range(len(feature_data)), feature_data)
            ax.set_title(f'Biểu đồ về {selected_data_item} ở cụm {selected_clus_item}')
            ax.set_xlabel('Số liệu')
            ax.set_ylabel('Gía trị')
            canvas = FigureCanvasTkAgg(fig, master=frame_chart_calcu)  # frame_chart_calcu is the parent widget
            canvas.draw()
            canvas.get_tk_widget().pack(side='top', fill='both', expand=True)
        else:
            print("No data to plot. Please select a field and cluster first.")



# Tạo giao diện

    # Phần 1:
    frame_P1 = ttk.Frame(frame, borderwidth=0, relief="solid", width=1400, height=350)
    frame_P1.pack(side='top', expand=True, fill='both')
    frame_P1.pack_propagate(False)

    # P1_1
    frame_upload_and_checkbox = ttk.Frame(frame_P1, borderwidth=0, relief="flat", width=400, height=320)
    frame_upload_and_checkbox.pack(side='left', padx=2, pady=2)
    frame_upload_and_checkbox.pack_propagate(False) 

    # Phần tải lên và checkbox
    frame_label_and_upload = ttk.Frame(frame_upload_and_checkbox, borderwidth=0, relief="flat", width=396, height=100)
    frame_label_and_upload.pack(side="top", pady=1)
    frame_label_and_upload.pack_propagate(False) 

    frame_upload = ttk.Frame(frame_label_and_upload, borderwidth=0, relief="flat")
    frame_upload.pack(side="top")

    label = ttk.Label(frame_upload, text=" ", width=40, borderwidth=1, relief="solid")
    label.pack(side='right', expand=True, padx=1, ipady=5)

    upload_button = ttk.Button(frame_upload, text="Upload file", command=handle_upload, width=24)
    upload_button.pack(side='left', expand=True, fill="both", padx=1, ipady=8)
    
    frame_tienxuly = ttk.Frame(frame_label_and_upload, borderwidth=0, relief="flat", width=150, height=60)
    frame_tienxuly.pack(side="top", padx=1, ipady=8)

    preprocess_button = ttk.Button(frame_tienxuly, text="Thông tin dữ liệu", command=preprocess, width=18)
    preprocess_button.pack(side='left',  fill='both', expand=True)
    
    prep_button = ttk.Button(frame_tienxuly, text="Tiền xử lý", command=prep, width=18)
    prep_button.pack(side='left', fill='both', expand=True)

    clear_button = ttk.Button(frame_tienxuly, text="Clear",command=clear_text, width=18)
    clear_button.pack(side='left', expand=True, fill="both")


    # Tạo frame cho phần chọn trường và số cụm
    frame_field_and_cluster = ttk.Frame(frame_upload_and_checkbox, borderwidth=1, relief="solid")
    frame_field_and_cluster.pack(side="top", fill="x")  # Sử dụng fill="x" để chiếm toàn bộ chiều ngang

    # Phần chọn trường
    frame_checkbox = tk.Frame(frame_field_and_cluster, borderwidth=1, relief="solid", width=238, height=205)
    frame_checkbox.pack(side="left", padx=1, pady=1)  # Sử dụng side="left" để hiển thị cạnh nhau
    frame_checkbox.pack_propagate(False)
    
    label_ct = ttk.Label(frame_checkbox, text='Chọn trường:')
    label_ct.pack(side='top')
    
    
# P1_2_1
    # frame_kmean_text = ttk.Frame(frame, borderwidth=1, relief="solid", width=780, height=209)
    # frame_kmean_text.pack(side="left", padx=1, pady=1)
    # frame_kmean_text.pack_propagate(False)

    # frame_scroll = ttk.Frame(frame_kmean_text, borderwidth=0, relief="flat", width=758, height=208)
    # frame_scroll.pack(side="left", padx=1, pady=1)
    # frame_scroll.pack_propagate(False)
    # text_1 = tk.Text(frame_scroll)
    # text_1.pack(side='top', fill="both", expand=True)

    # scrollbar_y = tk.Scrollbar(frame_kmean_text, orient="vertical", command=text_1.yview)
    # scrollbar_y.pack(side="right", fill="y")
    # text_1.configure(yscrollcommand=scrollbar_y.set)


    # Phần số cụm
    frame_kmean_cluster = ttk.Frame(frame_field_and_cluster, borderwidth=0, relief="flat", width=158, height=209)
    frame_kmean_cluster.pack(side="left", padx=1, pady=1) 
    frame_kmean_cluster.pack_propagate(False)

    # P1_2_2_1
    frame_kmean_cluster_text = ttk.Frame(frame_kmean_cluster, borderwidth=0, relief="solid", width=152, height=120)
    frame_kmean_cluster_text.pack(side="top", padx=1, pady=1)
    frame_kmean_cluster_text.pack_propagate(False)

    frame_tk = ttk.Frame(frame_kmean_cluster_text, borderwidth=0, relief="flat", width=144, height=60)
    frame_tk.pack(side="top", padx=1, pady=1)
    frame_tk.pack_propagate(False)

    label12 = ttk.Label(frame_tk, text='Số cụm')
    label12.pack(side='left', padx=1, pady=1)

    text_clus = tk.Text(frame_tk, width=10, height=1.5)
    text_clus.pack(side='left', padx=1, pady=1)

    frame_bk = ttk.Frame(frame_kmean_cluster_text, borderwidth=0, relief="flat", width=148, height=60)
    frame_bk.pack(side="top", padx=1, pady=1)
    frame_bk.pack_propagate(False)

    button_kmean = ttk.Button(frame_bk, text='Kmean', command=cluster_data)
    button_kmean.pack(side="left", fill="both")

    button_cluster = ttk.Button(frame_bk, text='Cluster', command=Elbow)
    button_cluster.pack(side="left", fill="both")

    # P1_2_2_2
    frame_export_kmean = ttk.Frame(frame_kmean_cluster, borderwidth=0, relief="flat", width=155, height=80)
    frame_export_kmean.pack(side="top", padx=1, pady=1)
    frame_export_kmean.pack_propagate(False)

    label_export_file = ttk.Label(frame_export_kmean, text='Xuất file:')
    label_export_file.pack(side='top', padx=1, pady=1)

    button_export_file = ttk.Button(frame_export_kmean, text='File', command=export_data)
    button_export_file.pack(side="top", padx=0, pady=1, ipadx=60, ipady=50)

    
 
    
    # Tạo frame cho Treeview và thanh cuộn
    frame_tree = ttk.Frame(frame_P1, borderwidth=0, relief="solid", width=1100, height=500)
    frame_tree.pack(side='right', padx=2, pady=2, fill='both', expand=True)

    # Khởi tạo Treeview
    tree = ttk.Treeview(frame_tree)
    tree.pack(side="top", fill="both", expand=True)

    # # Tạo thanh cuộn dọc cho Treeview
    # scrollbar_y = ttk.Scrollbar(frame_tree, orient="vertical", command=tree.yview)
    # scrollbar_y.pack(side="right", fill="y")
    # tree.configure(yscrollcommand=scrollbar_y.set)

    # Tạo thanh cuộn ngang cho Treeview
    scrollbar_x = ttk.Scrollbar(frame_tree, orient="horizontal", command=tree.xview)
    scrollbar_x.pack(side="bottom", fill="x")
    tree.configure(xscrollcommand=scrollbar_x.set)


    # prep_button = ttk.Button(frame_tienxuly, text="Tiền xử lý", command=prep)
    # prep_button.pack(side='left', ipadx=62, ipady=10, fill='both', expand=True)
#
    # scrollbar_y = tk.Scrollbar(frame_upload_and_checkbox, orient="vertical", command=tree.yview)
    # scrollbar_y.pack(side="right", fill="y")
    # scrollbar_x = tk.Scrollbar(frame_P1, orient="horizontal", command=tree.xview)
    # scrollbar_x.pack(side="top", fill="x")
    
    # # Liên kết thanh cuộn dọc với Treeview
    # tree.configure(yscrollcommand=scrollbar_y.set)

    # # Liên kết thanh cuộn ngang với Treeview
    # tree.configure(xscrollcommand=scrollbar_x.set)
    
    
    #----------------------------------------------------------------
    # Phần 2:
    frame_P2 = ttk.Frame(frame, borderwidth=0, relief="solid", width=1400, height=218)
    frame_P2.pack(side='top', expand=True, fill='both')
    frame_P2.pack_propagate(False)

    # P2_1
    frame_chart_kmean_kh = ttk.Frame(frame_P2, borderwidth=0, relief="solid", width=700, height=215)
    frame_chart_kmean_kh.pack(side="left", padx=3, pady=2, expand=True, fill='both')
    frame_chart_kmean_kh.pack_propagate(False)

    # P2_1_1
    frame_chart_kmean = ttk.Frame(frame_chart_kmean_kh, borderwidth=0, relief="flat", width=618, height=183)
    frame_chart_kmean.pack(side="top", padx=3, pady=3, expand=True, fill='both')
    frame_chart_kmean.pack_propagate(False)

    chart_kmean_button = ttk.Button(frame_chart_kmean_kh, text="Vẽ biểu đồ Kmean", command=plot_clustered_data)
    chart_kmean_button.pack(side="left", expand=True, pady=1)

    chart_show_kmean_button = ttk.Button(frame_chart_kmean_kh, text="Hiển thị biểu đồ Kmean", command=show_plot_clustered_data)
    chart_show_kmean_button.pack(side="left", expand=True, pady=1)


    # P2_2
    frame_chart_calcu_kh = ttk.Frame(frame_P2, borderwidth=0, relief="solid", width=700, height=215)
    frame_chart_calcu_kh.pack(side='left', padx=3, pady=2, expand=True, fill='both')
    frame_chart_calcu_kh.pack_propagate(False)

    # P2_2_1
    frame_chart_calcu = ttk.Frame(frame_chart_calcu_kh, borderwidth=0, relief="flat", width=618, height=183)
    frame_chart_calcu.pack(side="top", padx=3, pady=3, expand=True, fill='both')
    frame_chart_calcu.pack_propagate(False)

    chart_calcu_button = ttk.Button(frame_chart_calcu_kh, text="Vẽ biểu đồ thống kê", command=plot_statistics_value)
    chart_calcu_button.pack(side="left", expand=True, pady=1)

    chart_show_calcu_button = ttk.Button(frame_chart_calcu_kh, text="Hiển thị biểu đồ thống kê", command=show_plot_statistics_value)
    chart_show_calcu_button.pack(side="left", expand=True, pady=1)






    # Phần 3: 
    frame_P3 = ttk.Frame(frame, borderwidth=0, relief="flat", width=1400, height=218)
    frame_P3.pack(side='top', expand=True, fill='both')  # Cho phép mở rộng trong khung cha
    frame_P3.pack_propagate(False)

    # P3_1
    frame_combobox_check = ttk.Frame(frame_P3, borderwidth=0, relief="solid", width=300, height=210)
    frame_combobox_check.pack(side='left', padx=2, pady=2, fill='y')  # Chỉ cần giữ chiều cao cố định
    frame_combobox_check.pack_propagate(False)

    label_calcu_tk_kmean = ttk.Label(frame_combobox_check, text='Chọn trường, chọn trường tính thống kê', font=('Helvetica', 8, 'bold'))
    label_calcu_tk_kmean.pack(side='top', padx=5, pady=6, expand=True)

    # Frame chọn trường
    frame_fields = ttk.Frame(frame_combobox_check)
    frame_fields.pack(side='top', pady=3)
    label_fields = ttk.Label(frame_fields, text='Chọn Trường')
    label_fields.grid(row=0, column=0, padx=5, pady=5)

    combo_box_field = ttk.Combobox(frame_fields)
    combo_box_field.grid(row=0, column=1, padx=5, pady=5, ipady=5)
    combo_box_field.bind("<<ComboboxSelected>>", select_data_field)

    # Frame chọn cụm
    frame_clus = ttk.Frame(frame_combobox_check)
    frame_clus.pack(side="top", pady=3)

    label_clus = ttk.Label(frame_clus, text='Chọn Cụm')
    label_clus.grid(row=1, column=0, padx=10, pady=5)

    combo_box_cluster = ttk.Combobox(frame_clus)
    combo_box_cluster.grid(row=1, column=1, padx=10, pady=5, ipady=5)
    combo_box_cluster.bind("<<ComboboxSelected>>", select_clus_data)

    # P3_2: Hiển thị kết quả tính toán
    frame_text_calcu = ttk.Frame(frame_P3, borderwidth=0, relief="solid", width=1000, height=210)
    frame_text_calcu.pack(side='left', padx=2, pady=2, expand=True, fill='both')  # Mở rộng chiếm phần còn lại
    frame_text_calcu.pack_propagate(False)

    frame_text_1 = ttk.Frame(frame_text_calcu, borderwidth=0, relief="solid", width=971, height=200)
    frame_text_1.pack(side='left', padx=2, pady=2, expand=True, fill='both')  # Tương tự
    frame_text_1.pack_propagate(False)

    display_calcu_result = tk.Text(frame_text_1)
    display_calcu_result.pack(side='top', fill='both', expand=True)

    scrollbar_y = tk.Scrollbar(frame_text_calcu, orient="vertical", command=display_calcu_result.yview)
    scrollbar_y.pack(side="right", fill="y")
    display_calcu_result.configure(yscrollcommand=scrollbar_y.set)
