from setuptools import setup, find_packages

setup(
    name="yolospine",
    version="1.0.0",
    description=(
        "YOLOspine: A Clinically Validated Hierarchical Two-Stage "
        "Attention-Enhanced Architecture for Multi-Label Overlapping "
        "Spinal Disorder Detection"
    ),
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Rao Farhat Masood",
    author_email="farhatmasood.fm@gmail.com",
    url="https://github.com/farhatmasood/YOLOspine",
    license="MIT",
    python_requires=">=3.10",
    packages=find_packages(exclude=["tests", "docs", "assets"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "albumentations>=1.3.1",
        "opencv-python>=4.7.0",
        "Pillow>=9.5.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "scikit-learn>=1.2.0",
        "matplotlib>=3.7.0",
    ],
    extras_require={
        "ultralytics": ["ultralytics>=8.1.0"],
        "baselines": [
            "segmentation-models-pytorch>=0.3.3",
            "timm>=0.9.0",
        ],
        "all": [
            "ultralytics>=8.1.0",
            "segmentation-models-pytorch>=0.3.3",
            "timm>=0.9.0",
            "seaborn>=0.12.0",
            "scikit-image>=0.20.0",
            "pandas>=2.0.0",
        ],
    },
)
