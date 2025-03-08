"""
Utility functions for ColumnsGPT.
"""

import random
import pandas as pd
from typing import List, Dict, Any


def sample_dataframe(
    df: pd.DataFrame, column_name: str, sample_size: int = 10
) -> List[str]:
    """
    Take a random sample of values from a DataFrame column and convert to strings for analysis.

    Args:
        df: The pandas DataFrame to sample from
        column_name: The name of the column to sample
        sample_size: The number of samples to take (default: 10)

    Returns:
        A list of string representations of the sampled values
    """
    # Handle if the column has fewer entries than the sample size
    actual_sample_size = min(sample_size, len(df))

    # Get a random sample of indices
    if actual_sample_size == len(df):
        sample_indices = range(len(df))
    else:
        sample_indices = random.sample(range(len(df)), actual_sample_size)

    # Extract the values and convert to strings
    samples = [str(df.iloc[i][column_name]) for i in sample_indices]

    return samples


def determine_type_match(inferred_type: str, expected_type: str) -> bool:
    """
    Determine if the inferred type matches the expected type.
    This function handles various ways to express the same type.

    Args:
        inferred_type: The type inferred by the LLM
        expected_type: The expected type from the template

    Returns:
        True if the types match, False otherwise
    """
    # Normalize types to lowercase
    inferred = inferred_type.lower().strip()
    expected = expected_type.lower().strip()

    # Define type equivalence groups
    type_equivalents = {
        "int": ["int", "integer", "int64", "int32", "numpy.int64", "numpy.int32"],
        "float": [
            "float",
            "double",
            "float64",
            "float32",
            "numpy.float64",
            "numpy.float32",
        ],
        "str": ["str", "string", "text", "object"],
        "bool": ["bool", "boolean"],
        "date": ["date", "datetime", "timestamp", "pd.datetime", "pandas.datetime"],
        "category": ["category", "categorical"],
        "timedelta": ["timedelta", "time delta", "duration"],
    }

    # Check if types are directly equal
    if inferred == expected:
        return True

    # Check if types are in the same equivalence group
    for group, types in type_equivalents.items():
        if inferred in types and expected in types:
            return True

    return False


def format_analysis_results(
    summary: 'AnalysisSummary', rename_dict: Dict[str, str] = None
) -> str:
    """
    Format the analysis results into a readable string.

    Args:
        summary: The AnalysisSummary object
        rename_dict: Optional dictionary mapping DataFrame columns to template columns

    Returns:
        A formatted string with the analysis results
    """
    result = f"""
    # DataFrame Analysis Summary
    
    Total DataFrame columns: {summary.total_df_columns}
    Total template columns: {summary.total_template_columns}
    Analyzed columns: {summary.analyzed_columns}
    Matching columns: {summary.matching_columns}
    Non-matching columns: {summary.non_matching_columns}
    Unmatched DataFrame columns: {summary.unmatched_df_columns}
    Unmatched template columns: {summary.unmatched_template_columns}
    
    ## Detailed Column Analysis
    """

    # Add rename dictionary section if provided
    if rename_dict:
        result += f"""
    ## Rename Dictionary for DataFrame
    
    The following rename dictionary can be applied to transform your DataFrame to match the template:
    ```python
    rename_dict = {rename_dict}
    df_renamed = df.rename(columns=rename_dict)
    ```
    """

    # Add detailed analysis for each column
    for analysis in summary.column_analyses:
        match_status = (
            "Matches template"
            if analysis.matches_template
            else "Does not match template"
        )
        match_confidence = (
            f" (Match confidence: {analysis.match_confidence:.2f})"
            if analysis.match_confidence
            else ""
        )

        result += f"""
        ### {analysis.df_column_name} → {analysis.template_column_name or 'No match'}{match_confidence}
        - Inferred type: {analysis.inferred_type}
        - Confidence: {analysis.confidence:.2f}
        - Template type: {analysis.template_type or "N/A"}
        - Status: {match_status}
        - Sample values: {', '.join(analysis.sample_values[:3])}...
        - Notes: {analysis.notes or "N/A"}
        """

    return result