#!/usr/bin/env python
"""
Setup configuration for CausalTorch package.
"""

from setuptools import setup, find_packages

# Read long description from README.md
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = (
        "CausalTorch: A PyTorch library for building generative models with explicit causal constraints."
    )

setup(
    name="causaltorch",
    version="0.1.0",
    author="Elija Nzeli",
    author_email="elijahnzeli@example.com",  # Replace with your actual email
    description="Causal Neural-Symbolic Generative Networks for PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/elijahnzeli1/CausalTorch",
    project_urls={
        "Bug Tracker": "https://github.com/elijahnzeli1/CausalTorch/issues",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.8.0",
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "networkx>=2.5",
    ],
    extras_require={
        "text": ["transformers>=4.5.0"],
        "dev": ["pytest>=6.0.0", "black>=21.5b2", "isort>=5.8.0"],
    },
)