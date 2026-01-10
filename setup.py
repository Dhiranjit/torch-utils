from setuptools import setup, find_packages

setup(
    name="torch-utils",
    version="0.1.0",
    description="Lightweight reusable PyTorch utilities for deep learning projects",
    author="Dhiranjit",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "matplotlib"
    ],
)
