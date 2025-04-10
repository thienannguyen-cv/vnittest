# vnittest/web_app.py
import os
import subprocess
import pkg_resources
import webbrowser
import time
import threading

def open_browser(port=8866):
    """Mở trình duyệt sau khi server đã khởi động"""
    # Đợi server khởi động
    time.sleep(2)
    webbrowser.open(f"http://localhost:{port}")

def run_app():
    """Khởi chạy ứng dụng web dựa trên Voilà"""
    # Đường dẫn tới notebook
    notebook_path = pkg_resources.resource_filename('vnittest', 'vnittest.ipynb')
    
    # Kiểm tra xem notebook có tồn tại không
    if not os.path.exists(notebook_path):
        print(f"Error: Notebook not found at {notebook_path}")
    
    # Khởi động trình duyệt web trong một luồng riêng biệt
    port = 8866
    threading.Thread(target=open_browser, args=(port,)).start()
    
    # Khởi động Voilà server
    print(f"Starting web app on http://localhost:{port}...")
    subprocess.run([
        "voila", 
        notebook_path,
        "--port", str(port),
        "--theme=light"
    ])