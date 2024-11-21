import tkinter as tk
import os
from tkinter import ttk 
from tab_1 import upload_and_display_file
# from tab_2 import Phankhuc
from tab_3 import kmean
from tab_4 import create_help_tab

# current_dir = os.path.dirname(os.path.abspath(__file__))
# icon_path = os.path.join(current_dir, "img", "Logo.ico")

# Giao diện            
window = tk.Tk()
window.title("Phân cụm dữ liệu khách hàng bằng thuật toán Kmean")
# window.iconbitmap(icon_path)
window.geometry("1400x750")
# window.resizable(width=False, height=False)
notebook = ttk.Notebook(window)

# Tạo khung chính
frame = ttk.Frame(window)
frame.pack(fill='both', expand=True)  # Khung chiếm toàn bộ không gian


# Tab 1
upload_tab = ttk.Frame(notebook)
notebook.add(upload_tab, text='Phân Khúc Khách Hàng')
upload_and_display_file(upload_tab)

# Tab 2
# segment_tab = ttk.Frame(notebook)
# notebook.add(segment_tab, text='Phân Khúc Khách Hàng')
# Phankhuc(segment_tab, upload_tab)

# Tab 3
kmean_tab = ttk.Frame(notebook)
notebook.add(kmean_tab, text="Kmean")
kmean(kmean_tab)


help_tab = ttk.Frame(notebook)
notebook.add(help_tab, text="Hướng dẫn")
create_help_tab(help_tab)

# Chạy file
notebook.pack(fill='both', expand=True)

window.mainloop()
