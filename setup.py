# setup.py
from setuptools import setup, find_packages
import os, re

def read_version():
    with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "vnittest/__init__.py"), "r", encoding="utf8") as f:
        content = f.read()
    version_match = re.search(r"^__version__\s*=\s*['\"]([^'\"]+)['\"]", content, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Could not find __version__ in __init__.py")

# Đọc file requirements.txt và loại bỏ các dòng trống hoặc comment
with open("requirements.txt", "r", encoding="utf-8") as req_file:
    install_requires = [
        line.strip() for line in req_file
        if line.strip() and not line.startswith("#")
    ]
    
# Đọc nội dung README.md
with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name='vnittest',
    version=read_version(),
    packages=find_packages(),
    install_requires=install_requires,
    entry_points={
        'console_scripts': [
            'vnittest = vnittest.web_app:run_app'
        ]
    },
    author='Thien An L. Nguyen',
    author_email="thienannguyen.cv@gmail.com",
    url="https://github.com/thienannguyen-cv/vnittest",
    description='UI interface for gradient flow visualization',
    long_description=long_description,
    long_description_content_type="text/markdown",  # hoặc "text/x-rst"
    license='Apache 2.0',
    # Thêm các tệp dữ liệu nếu cần
    include_package_data=True,
    package_data={
        'vnittest': ['notebooks/*.ipynb'],
    },
)
