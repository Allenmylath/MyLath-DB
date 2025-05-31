# setup.py
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mylath",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A graph database with vector search using Redis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mylath",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "redis>=4.0.0",
        "numpy>=1.21.0",
        "flask>=2.0.0",
        "click>=8.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "isort>=5.0",
            "flake8>=3.8",
            "mypy>=0.910",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
        "graphblas": [
            "python-graphblas>=2022.4.5",
        ],
    },
    entry_points={
        "console_scripts": [
            "mylath-server=mylath.cli:main",
        ],
    },
)