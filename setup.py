"""
Setup script for GeoGPT
"""

import os
from setuptools import setup, find_packages

# Read the contents of README file
with open(os.path.join(os.path.dirname(__file__), "README.md"), "r") as f:
    readme = f.read()

# Define package requirements
import sys

if sys.version_info >= (3, 10):
    pgeocode_req = "pgeocode>=0.6.0"
else:
    pgeocode_req = "pgeocode>=0.4.1,<0.5.0"

requirements = [
    pgeocode_req,
    "pandas>=1.0.0",
    "numpy>=1.18.0",
    "langchain>=0.1.0",
    "langchain-core>=0.1.0",
    "langchain-openai>=0.0.1",
    "langchain-anthropic>=0.0.1",
    "langchain-google-genai>=0.0.1",
    "langchain-deepseek>=0.0.1",
    "python-dotenv>=0.19.0",
    "pycountry>=22.3.5",
]

setup(
    name="geo-gpt",
    version="0.1.6",
    description="Enhanced geocoding with pgeocode and LLM fallback",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Josh Janzen",
    author_email="joshjanzen@gmail.com",
    url="https://github.com/zen-apps/geo-gpt",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "geo_gpt": ["data/*.csv"],
    },
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "geo-gpt=geo_gpt.cli:main",
        ],
    },
)
