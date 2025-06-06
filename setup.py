"""
Setup script for Cypher Planner
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="cypher-planner",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A Cypher to logical execution plan converter for Redis + GraphBLAS architecture",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/cypher-planner",
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
        "Topic :: Database",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        # No core dependencies - uses only Python standard library
    ],
    extras_require={
        "redis": ["redis>=4.0.0"],
        "graphblas": ["python-graphblas>=2021.12.0", "scipy>=1.7.0"],
        "dev": [
            "pytest>=6.0.0",
            "black>=21.0.0",
            "flake8>=3.8.0",
            "mypy>=0.910",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "cypher-planner=main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
