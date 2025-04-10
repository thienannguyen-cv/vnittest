# setup.py
from setuptools import setup, find_packages

# Đọc file requirements.txt và loại bỏ các dòng trống hoặc comment
with open("requirements.txt", "r", encoding="utf-8") as req_file:
    install_requires = [
        line.strip() for line in req_file
        if line.strip() and not line.startswith("#")
    ]

setup(
    name='vnittest',
    version='0.0.0',
    packages=find_packages(),
    install_requires=install_requires,
    entry_points={
        'console_scripts': [
            'vnittest = vnittest.main:main'
        ]
    },
    author='Thien An L. Nguyen',
    description='UI interface for gradient flow visualization',
    license='Apache 2.0',
)