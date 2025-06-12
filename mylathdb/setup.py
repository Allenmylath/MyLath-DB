
from setuptools import setup, find_packages

setup(
    name="mylathdb",
    version="1.0.0",
    description="MyLathDB - Graph Database with Cypher Support",
    packages=find_packages(),
    install_requires=[
        "redis>=4.0.0",
        "numpy>=1.20.0",
        "psutil>=5.8.0",
    ],
    extras_require={
        "graphblas": ["graphblas>=2022.12.0"],
        "dev": ["pytest>=6.0.0", "pytest-cov>=2.0.0"],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
