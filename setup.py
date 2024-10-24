from pathlib import Path

from setuptools import setup

directory = Path(__file__).resolve().parent
with open(directory / "README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="triton-kernels",
    version="0.1.0",
    author="Sinjin Jeong",
    description="Triton kernels for SD3 and Flux",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ai-compiler-study/triton-kernels",
    packages=[
        "triton_kernels",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    install_requires=[
        "numpy",
        "torch",
        "triton>=2.2.0",
        "einops",
    ],
    extras_require={
        "linting": [
            "pre-commit>=3.5.0",
        ],
        "testing": [
            "pytest>=7.4.0",
            "pytest-xdist>=3.5.0",
        ],
    },
    python_requires=">=3.8",
)
