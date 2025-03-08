# Steps to Publish to PyPI

This document outlines the steps to prepare and publish the geo-gpt package to PyPI.

## Preparation

1. **Update Version Number**: 
   - Edit `setup.py` and update the version number

2. **Clean and validate the repository**:
   ```bash
   # Remove any build artifacts
   rm -rf build/ dist/ *.egg-info/
   
   # Run tests (if available)
   python -m unittest discover
   
   # Check package structure
   python setup.py check
   ```

3. **Install publishing tools**:
   ```bash
   pip install --upgrade pip setuptools wheel twine build
   ```

## Building the Package

1. **Build the distribution packages**:
   ```bash
   # Using build (recommended)
   python -m build
   
   # Alternative method
   python setup.py sdist bdist_wheel
   ```

2. **Check the built distribution**:
   ```bash
   twine check dist/*
   ```

## Testing in TestPyPI (Recommended)

1. **Upload to TestPyPI**:
   ```bash
   twine upload --repository-url https://test.pypi.org/legacy/ dist/*
   ```

2. **Install from TestPyPI to test**:
   ```bash
   pip install --index-url https://test.pypi.org/simple/ --no-deps geo-gpt
   
   # Then install dependencies separately
   pip install pgeocode pandas numpy langchain langchain-core langchain-openai python-dotenv
   ```

3. **Test the package**: Ensure it works as expected

## Publishing to PyPI

1. **Upload to PyPI**:
   ```bash
   twine upload dist/*
   ```

2. **Verify installation**:
   ```bash
   pip install geo-gpt
   ```

3. **Create a git tag for the release**:
   ```bash
   git tag -a v0.1.0 -m "Release v0.1.0"
   git push origin v0.1.0
   ```

## Notes

- You'll need PyPI credentials to upload the package.
- If this is your first time uploading, you might need to register on PyPI and TestPyPI.
- Make sure all your dependencies are correctly specified in setup.py.
- Remember to remove the `jupyter_notebook` directory before publishing.