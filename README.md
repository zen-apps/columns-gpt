# ColumnsGPT

DataFrame column analysis and matching with LLM

## Overview

ColumnsGPT is a Python package that leverages large language models (LLMs) to analyze and match columns in pandas DataFrames against template schemas. It helps data analysts and scientists automate the tedious task of mapping columns between different data sources, especially when dealing with varied naming conventions.

## Features

- Semantic column name matching using LLMs
- Data type inference and validation
- Support for multiple LLM providers (OpenAI, Google, Anthropic, DeepSeek)
- Interactive mode for user feedback on column mappings
- Detailed analysis reports with confidence scores
- Easy-to-use Python API and command-line interface

## Installation

```bash
pip install columns-gpt
```

## Requirements

- Python 3.8+
- pandas
- numpy
- langchain and related packages
- python-dotenv
- pydantic

## Setting Up Environment Variables (IMPORTANT)

ColumnsGPT requires environment variables to be set for LLM access. Follow these steps:

1. **Copy the template:** A template file is provided in the repository:
   ```bash
   cp .env.template .env
   ```

2. **Edit the file:** Open `.env` and add your API key for the LLM provider you want to use:
   ```
   LLM_PROVIDER=openai  # Choose: openai, google, anthropic, deepseek
   OPENAI_API_KEY=your-api-key-here
   LLM_MODEL_OPENAI=gpt-4o  # Optional, will use default if not specified
   ```

3. **Alternative:** Set environment variables directly:
   ```bash
   # For OpenAI
   export LLM_PROVIDER=openai
   export OPENAI_API_KEY=your-api-key-here
   export LLM_MODEL_OPENAI=gpt-4o
   
   # For Google
   export LLM_PROVIDER=google
   export GOOGLE_API_KEY=your-api-key-here
   export LLM_MODEL_GOOGLE=gemini-2.0-flash-exp
   
   # For Anthropic
   export LLM_PROVIDER=anthropic
   export ANTHROPIC_API_KEY=your-api-key-here
   export LLM_MODEL_ANTHROPIC=claude-3-5-sonnet-latest
   
   # For DeepSeek
   export LLM_PROVIDER=deepseek
   export DEEPSEEK_API_KEY=your-api-key-here
   export LLM_MODEL_DEEPSEEK=deepseek-chat
   ```

> **IMPORTANT**: Always set environment variables BEFORE importing the package.
> In production environments, use secure environment variable management rather than .env files.

## Basic Usage

```python
# IMPORTANT: Set up environment variables first
import os
import pandas as pd
from dotenv import load_dotenv

# Load environment variables (in development)
load_dotenv()  # Will look for .env in current directory

# Or set them directly (better for production)
os.environ["LLM_PROVIDER"] = "openai"
os.environ["OPENAI_API_KEY"] = "your-api-key-here"
os.environ["LLM_MODEL_OPENAI"] = "gpt-4o"  # Optional

# Create a sample DataFrame
data = {
    "user_id": [1, 2, 3, 4, 5],
    "full_name": ["Alice Johnson", "Bob Smith", "Charlie Brown", "David Miller", "Eve Davis"],
    "customer_age": [25, 30, 35, 40, 45],
    "annual_salary": [50000.0, 60000.0, 70000.0, 80000.0, 90000.0],
    "is_active": [True, True, False, True, False],
    "registration_date": ["2020-01-01", "2019-05-15", "2021-03-10", "2018-11-20", "2022-02-28"],
}
df = pd.DataFrame(data)

# Define a template with different column names but semantically similar concepts
template = {
    "transaction_id": "int",
    "athlete": "str", 
    "years_old": "int",
    "salary": "float",
    "employed": "bool",
    "begin_date": "date",
    "weight": "int",
}

# Import and use the package
from columns_gpt import analyze_dataframe
from columns_gpt.utils import format_analysis_results

# Analyze the DataFrame
analysis_results, summary, rename_dict = analyze_dataframe(
    df, template, llm_provider="openai", match_threshold=0.5
)

# Print formatted results
print(format_analysis_results(summary, rename_dict))

# Apply the rename dictionary to the DataFrame
df_renamed = df.rename(columns=rename_dict)
print(df_renamed.head())
```

## Command Line Interface

The command-line interface automatically attempts to load environment variables from a `.env` file in the current directory:

```bash
# Non-interactive analysis
columns-gpt --input data.csv --template template.json --provider openai --threshold 0.5 --output results.json

# Interactive analysis (allows you to adjust column mappings)
columns-gpt --input data.csv --template template.json --interactive
```

## API Reference

### Main Functions

```python
# Non-interactive analysis
analyze_dataframe(
    df, template, llm_provider="openai", sample_size=10, 
    match_threshold=0.7, max_columns=None, rename_mapping=None
)

# Interactive analysis with user feedback
interactive_analyze_dataframe(
    df, template, llm_provider="openai"
)
```

### Data Models

#### ColumnAnalysis

A data model representing analysis of a single DataFrame column.

**Attributes:**
- `df_column_name`: The name of the column in the DataFrame
- `template_column_name`: The matched template column name (if any)
- `inferred_type`: The inferred data type
- `confidence`: Confidence score for type inference (0-1)
- `sample_values`: Sample values from the column
- `matches_template`: Whether the column matches template requirements
- `template_type`: The expected type from the template
- `match_confidence`: Confidence score for column name match (0-1)
- `notes`: Additional observations

#### AnalysisSummary

A data model containing the overall analysis results.

**Attributes:**
- `total_df_columns`: Total number of columns in the DataFrame
- `total_template_columns`: Total number of columns in the template
- `analyzed_columns`: Number of columns analyzed
- `matching_columns`: Number of columns matching the template
- `non_matching_columns`: Number of columns not matching the template
- `unmatched_df_columns`: Number of DataFrame columns not matched to template
- `unmatched_template_columns`: Number of template columns not matched to DataFrame
- `column_analyses`: List of detailed column analyses

## Environment Variables

These environment variables configure ColumnsGPT's behavior:

### Required Variables

For using ColumnsGPT, you MUST set:

1. `LLM_PROVIDER`: Which LLM provider to use (required)
   - Options: `openai`, `google`, `anthropic`, `deepseek`

2. API key for your chosen provider (one of these is required):
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `GOOGLE_API_KEY`: Your Google API key
   - `ANTHROPIC_API_KEY`: Your Anthropic API key
   - `DEEPSEEK_API_KEY`: Your DeepSeek API key

### Optional Variables

3. Model name for your chosen provider (optional, defaults provided):
   - `LLM_MODEL_OPENAI`: OpenAI model (default: "gpt-4o")
   - `LLM_MODEL_GOOGLE`: Google model (default: "gemini-2.0-flash-exp")
   - `LLM_MODEL_ANTHROPIC`: Anthropic model (default: "claude-3-5-sonnet-latest")
   - `LLM_MODEL_DEEPSEEK`: DeepSeek model (default: "deepseek-chat")

## Examples

### Template Format

The template should be a JSON object with column names as keys and expected data types as values:

```json
{
  "user_id": "int",
  "name": "str",
  "age": "int",
  "salary": "float",
  "active": "bool",
  "start_date": "date"
}
```

### Interactive Analysis

```python
import pandas as pd
from columns_gpt import interactive_analyze_dataframe

# Load your DataFrame
df = pd.read_csv("data.csv")

# Define your template
template = {
    "id": "int",
    "full_name": "str",
    "age": "int",
    "income": "float",
    "is_employed": "bool",
    "registration_date": "date"
}

# Run interactive analysis
interactive_analyze_dataframe(df, template, llm_provider="openai")
```

## Troubleshooting

1. **API Key errors**: Make sure your API key is correct and has appropriate permissions

2. **Pydantic version issues**: This package uses Pydantic for data models. If you get errors related to Pydantic, try:
   ```bash
   pip install "pydantic>=2.0.0"
   ```

3. **LLM provider errors**: Check that you've set the correct environment variables for your chosen provider

4. **"No module named dotenv"**: Install python-dotenv if you're using .env files
   ```bash
   pip install python-dotenv
   ```

## License

MIT
