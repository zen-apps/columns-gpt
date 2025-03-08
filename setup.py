"""
Setup script for ColumnsGPT
"""

import os
from setuptools import setup, find_packages

# Read the contents of README file
with open(os.path.join(os.path.dirname(__file__), "README.md"), "r") as f:
    readme = f.read()

requirements = [
    "pandas>=1.0.0",
    "numpy>=1.18.0",
    "langchain>=0.1.0",
    "langchain-core>=0.1.0",
    "langchain-openai>=0.0.1",
    "langchain-anthropic>=0.0.1",
    "langchain-google-genai>=0.0.1",
    "langchain-deepseek>=0.0.1",
    "python-dotenv>=0.19.0",
    "pydantic>=2.0.0",
]

setup(
    name="columns-gpt",
    version="0.1.2",
    description="DataFrame column analysis and matching with LLM",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Josh Janzen",
    author_email="joshjanzen@gmail.com",
    url="https://github.com/zen-apps/columns-gpt",
    packages=find_packages(),
    include_package_data=True,
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
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "columns-gpt=columns_gpt.cli:main",
        ],
    },
)
