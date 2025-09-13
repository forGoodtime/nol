#!/usr/bin/env python3
"""
Setup script для AIinDrive - Vehicle Damage Detection System
"""
from setuptools import setup, find_packages
import os

# Читаем README для длинного описания
def read_readme():
    with open("README.md", "r", encoding="utf-8") as f:
        return f.read()

# Читаем requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="aiindrive",
    version="0.1.0",
    author="AIinDrive Team",
    author_email="team@aiindrive.com",
    description="AI-powered vehicle damage detection and analysis system",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/aiindrive",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.2.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.11.0",
            "flake8>=5.0.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "jupyterlab>=3.5.0",
            "ipykernel>=6.17.0",
            "ipywidgets>=8.0.0",
        ],
        "inference": [
            "fastapi>=0.95.0",
            "uvicorn>=0.20.0",
            "gradio>=3.13.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "aiindrive-train=src.trainers.trainer_classification:main",
            "aiindrive-infer=src.inference.infer:main",
            "aiindrive-prepare-data=scripts.prepare_data:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.md"],
        "src": ["config/*.yaml"],
    },
    zip_safe=False,
    keywords="computer-vision, vehicle-damage, deep-learning, pytorch, damage-detection",
    project_urls={
        "Bug Reports": "https://github.com/your-org/aiindrive/issues",
        "Source": "https://github.com/your-org/aiindrive",
        "Documentation": "https://github.com/your-org/aiindrive/wiki",
    },
)
