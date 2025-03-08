"""
Basic usage example for ColumnsGPT
"""

import pandas as pd
from columns_gpt import analyze_dataframe
from columns_gpt.utils import format_analysis_results

# Create a sample DataFrame with columns that don't match template names exactly
data = {
    "user_id": [1, 2, 3, 4, 5],
    "full_name": [
        "Alice Johnson",
        "Bob Smith",
        "Charlie Brown",
        "David Miller",
        "Eve Davis",
    ],
    "customer_age": [25, 30, 35, 40, 45],
    "annual_salary": [50000.0, 60000.0, 70000.0, 80000.0, 90000.0],
    "is_active": [True, True, False, True, False],
    "registration_date": [
        "2020-01-01",
        "2019-05-15",
        "2021-03-10",
        "2018-11-20",
        "2022-02-28",
    ],
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

# Analyze the DataFrame against the template
print("Analyzing DataFrame...")
analysis_results, summary, rename_dict = analyze_dataframe(
    df, template, llm_provider="openai", sample_size=10, match_threshold=0.5
)

# Print the formatted results
print(format_analysis_results(summary, rename_dict))

# Apply the rename dictionary to the DataFrame
if rename_dict:
    df_renamed = df.rename(columns=rename_dict)
    print("\nRenamed DataFrame:")
    print(df_renamed.head())