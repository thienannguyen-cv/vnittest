# vnittest/web_app.py
import os, sys
import subprocess
import pkg_resources
import webbrowser
import time
import threading
import argparse

def open_browser(port=8866):
    """Mở trình duyệt sau khi server đã khởi động"""
    # Đợi server khởi động
    time.sleep(2)
    webbrowser.open(f"http://localhost:{port}")

def run_app():
    """Khởi chạy ứng dụng web dựa trên Voilà"""
    # Đường dẫn tới notebook
    debug_path = "./tests/test_data/test_"
    port = 8866
    notebook_path = pkg_resources.resource_filename('vnittest', 'notebooks/vnittest.ipynb')
    
    parser = argparse.ArgumentParser(description="Run the VNITest gradient flow visualization web UI.")
    parser.add_argument("debug_path", nargs="?", default=debug_path,
                        help=f"Path to debug data or folder (default: {debug_path})")
    parser.add_argument("--port", type=int, default=8866,
                        help=f"Port to run the Voilà app (default: {port})")
    parser.add_argument("--notebook", type=str, default=notebook_path,
                        help=f"Path to Jupyter notebook {notebook_path}")
    
    args = parser.parse_args()
    debug_path = args.debug_path
    port = args.port
    notebook_path = args.notebook
    
    debug_path_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), "notebooks/debug_path")
    with open(debug_path_file, 'w', encoding='utf-8') as f:
        f.write(debug_path)
    
    # Kiểm tra xem notebook có tồn tại không
    if not os.path.exists(notebook_path):
        print(f"Error: Notebook not found at {notebook_path}")
    if not os.path.exists(debug_path_file):
        print(f"Error: Cannot access debug path")
    
    # Dùng Popen để không chặn kernel
    subprocess.Popen([
        "voila", 
        notebook_path,
        "--port", str(port),
        "--theme=light",
        "--no-browser"
    ])

    open_browser(port)