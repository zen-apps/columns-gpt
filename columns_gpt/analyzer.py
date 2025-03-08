"""
Core functionality for DataFrame column analysis and matching.
"""

import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Set
from langchain.prompts import PromptTemplate

from .models import (
    ColumnAnalysis,
    ColumnMatchCandidate,
    ColumnsMatchResponse,
    AnalysisSummary,
)
from .llm import return_llm
from .prompts import get_column_analysis_prompt, get_column_matching_prompt
from .utils import sample_dataframe, determine_type_match, format_analysis_results


def match_columns(
    df: pd.DataFrame,
    template: Dict[str, str],
    llm_provider: str = "openai",
    match_threshold: float = 0.7,
) -> List[ColumnMatchCandidate]:
    """
    Use the LLM to match DataFrame columns to template columns based on semantic similarity.

    Args:
        df: The pandas DataFrame containing the columns to match
        template: A dictionary where keys are column names and values are expected data types
        llm_provider: The LLM provider to use for matching
        match_threshold: Minimum confidence threshold for considering a match (default: 0.7)

    Returns:
        A list of ColumnMatchCandidate objects with the matching results
    """
    # Get column samples for context
    df_samples = {}
    for col in df.columns:
        samples = sample_dataframe(df, col, sample_size=3)
        df_samples[col] = samples

    # Format sample data for the prompt
    df_samples_str = "\n".join(
        [f"- {col}: {', '.join(samples)}" for col, samples in df_samples.items()]
    )

    # Create the prompt
    prompt_template = PromptTemplate(
        template=get_column_matching_prompt(),
        input_variables=["df_columns", "template_columns", "df_samples"],
    )

    prompt_filled_in = prompt_template.format(
        df_columns=", ".join(df.columns),
        template_columns="\n".join(
            [f"- {col}: {dtype}" for col, dtype in template.items()]
        ),
        df_samples=df_samples_str,
    )

    # Initialize the LLM
    llm = return_llm(llm_provider)
    llm_with_structured_output = llm.with_structured_output(ColumnsMatchResponse)

    # Get the matching result
    result = llm_with_structured_output.invoke(prompt_filled_in)

    # Filter matches by threshold
    filtered_matches = [
        match for match in result.matches if match.match_confidence >= match_threshold
    ]

    return filtered_matches


def analyze_column(
    df: pd.DataFrame,
    df_column_name: str,
    template_column_name: Optional[str] = None,
    expected_type: Optional[str] = None,
    match_confidence: Optional[float] = None,
    llm_provider: str = "openai",
) -> ColumnAnalysis:
    """
    Analyze a single column using the specified LLM provider.

    Args:
        df: The pandas DataFrame containing the column
        df_column_name: The name of the column in the DataFrame to analyze
        template_column_name: The matched template column name (if any)
        expected_type: The expected data type from the template (if provided)
        match_confidence: The confidence score for the column name match (if any)
        llm_provider: The LLM provider to use for analysis

    Returns:
        A ColumnAnalysis object with the analysis results
    """
    # Sample the DataFrame column
    samples = sample_dataframe(df, df_column_name)

    # Create the prompt
    prompt_template = PromptTemplate(
        template=get_column_analysis_prompt(),
        input_variables=["column_name", "sample_values", "expected_type"],
    )

    prompt_filled_in = prompt_template.format(
        column_name=df_column_name,
        sample_values="\n".join([f"- {sample}" for sample in samples]),
        expected_type=expected_type if expected_type else "Not specified",
    )

    # Initialize the LLM
    llm = return_llm(llm_provider)
    llm_with_structured_output = llm.with_structured_output(ColumnAnalysis)

    # Get the analysis result
    result = llm_with_structured_output.invoke(prompt_filled_in)

    # Set the template match status if there's an expected type
    if expected_type:
        result.matches_template = determine_type_match(
            result.inferred_type, expected_type
        )
        result.template_type = expected_type

    # Set the DataFrame and template column names
    result.df_column_name = df_column_name
    result.template_column_name = template_column_name
    result.match_confidence = match_confidence

    return result


def analyze_dataframe(
    df: pd.DataFrame,
    template: Dict[str, str],
    llm_provider: str = "openai",
    sample_size: int = 10,
    match_threshold: float = 0.7,
    max_columns: Optional[int] = None,
    rename_mapping: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, ColumnAnalysis], AnalysisSummary, Dict[str, str]]:
    """
    Analyze a DataFrame against a template using the specified LLM provider.

    Args:
        df: The pandas DataFrame to analyze
        template: A dictionary where keys are column names and values are expected data types
        llm_provider: The LLM provider to use for analysis
        sample_size: The number of samples to take from each column
        match_threshold: Minimum confidence threshold for considering a match (default: 0.7)
        max_columns: Maximum number of columns to analyze (for cost/time constraints)
        rename_mapping: Optional dictionary to override automatic column mapping (df_column -> template_column)

    Returns:
        A tuple containing:
            1. A dictionary mapping DataFrame column names to their analysis results
            2. A summary of the analysis results
            3. A rename dictionary that can be applied to the DataFrame (df_column -> template_column)
    """
    # Step 1: Match DataFrame columns to template columns (unless mapping is provided)
    if rename_mapping is None:
        column_matches = match_columns(df, template, llm_provider, match_threshold)

        # Create a mapping of DataFrame columns to template columns
        df_to_template_map = {}
        for match in column_matches:
            df_to_template_map[match.df_column_name] = (
                match.template_column_name,
                match.match_confidence,
            )
    else:
        # Use the provided mapping
        df_to_template_map = {
            df_col: (template_col, 1.0)  # Use confidence of 1.0 for manual mappings
            for df_col, template_col in rename_mapping.items()
            if df_col in df.columns and template_col in template
        }

    # Determine which DataFrame columns to analyze
    columns_to_analyze = set(df_to_template_map.keys())

    # If max_columns is specified, limit the number of columns
    if max_columns and len(columns_to_analyze) > max_columns:
        columns_to_analyze = set(list(columns_to_analyze)[:max_columns])

    # Initialize result dictionary
    column_analyses = {}

    # Analyze each matched column
    for df_column_name in columns_to_analyze:
        template_column_name, match_confidence = df_to_template_map[df_column_name]
        expected_type = template.get(template_column_name)

        column_analyses[df_column_name] = analyze_column(
            df,
            df_column_name,
            template_column_name,
            expected_type,
            match_confidence,
            llm_provider,
        )

    # Create analysis summary
    matching_columns = sum(
        1 for analysis in column_analyses.values() if analysis.matches_template
    )
    non_matching_columns = sum(
        1 for analysis in column_analyses.values() if analysis.matches_template is False
    )

    # Calculate unmatched columns
    matched_df_columns = set(column_analyses.keys())

    # Get matched template columns correctly
    matched_template_columns = set()
    for df_col in df_to_template_map:
        if df_col in column_analyses:
            template_col = df_to_template_map[df_col][0]
            matched_template_columns.add(template_col)

    unmatched_df_columns = len(set(df.columns) - matched_df_columns)
    unmatched_template_columns = len(set(template.keys()) - matched_template_columns)

    summary = AnalysisSummary(
        total_df_columns=len(df.columns),
        total_template_columns=len(template),
        analyzed_columns=len(column_analyses),
        matching_columns=matching_columns,
        non_matching_columns=non_matching_columns,
        unmatched_df_columns=unmatched_df_columns,
        unmatched_template_columns=unmatched_template_columns,
        column_analyses=list(column_analyses.values()),
    )

    # Create a rename dictionary for easy application to the DataFrame
    rename_dict = {}
    for analysis in summary.column_analyses:
        if analysis.template_column_name and analysis.matches_template:
            # Only include mappings where the template column actually exists in the template
            if analysis.template_column_name in template:
                rename_dict[analysis.df_column_name] = analysis.template_column_name

    return column_analyses, summary, rename_dict


def get_user_feedback_on_mappings(
    df: pd.DataFrame,
    template: Dict[str, str],
    column_matches: List[ColumnMatchCandidate],
) -> Dict[str, str]:
    """
    Present column mappings to the user and get their feedback.

    Args:
        df: The pandas DataFrame containing the columns
        template: A dictionary where keys are column names and values are expected data types
        column_matches: List of ColumnMatchCandidate objects with potential matches

    Returns:
        A dictionary mapping DataFrame columns to template columns based on user feedback
    """
    print("\nSuggested Column Mappings:")
    print("=========================")

    # Create a mapping from DataFrame columns to template columns
    mapping = {}
    for match in column_matches:
        # Only include matches to columns that exist in the template
        if match.template_column_name in template and match.match_confidence >= 0.7:
            mapping[match.df_column_name] = match.template_column_name

    # Display current mapping
    for i, (df_col, template_col) in enumerate(mapping.items(), 1):
        print(
            f"{i}. {df_col} → {template_col} (Confidence: {next((m.match_confidence for m in column_matches if m.df_column_name == df_col and m.template_column_name == template_col), 0):.2f})"
        )

    # Display unmapped DataFrame columns
    unmapped_df_cols = set(df.columns) - set(mapping.keys())
    if unmapped_df_cols:
        print("\nUnmapped DataFrame columns:")
        for col in unmapped_df_cols:
            print(f"- {col}")

    # Display unmapped template columns
    unmapped_template_cols = set(template.keys()) - set(mapping.values())
    if unmapped_template_cols:
        print("\nUnmapped template columns:")
        for col in unmapped_template_cols:
            print(f"- {col} ({template[col]})")

    # Get user feedback
    print("\nDo you want to adjust any mappings? (y/n)")
    adjust = input().lower().strip()

    if adjust == "y":
        while True:
            print("\nEnter an adjustment (or 'done' when finished):")
            print(
                "Format: 'df_column_name:template_column_name' or 'remove:df_column_name' or 'done'"
            )

            user_input = input().strip()
            if user_input.lower() == "done":
                break

            if user_input.startswith("remove:"):
                # Remove a mapping
                df_col = user_input.split(":", 1)[1].strip()
                if df_col in mapping:
                    del mapping[df_col]
                    print(f"Removed mapping for {df_col}")
                else:
                    print(f"No mapping found for {df_col}")
            elif ":" in user_input:
                # Add or update a mapping
                df_col, template_col = user_input.split(":", 1)
                df_col = df_col.strip()
                template_col = template_col.strip()

                if df_col not in df.columns:
                    print(f"Warning: {df_col} is not in the DataFrame columns.")
                    continue

                if template_col not in template:
                    print(f"Warning: {template_col} is not in the template columns.")
                    continue

                mapping[df_col] = template_col
                print(f"Updated mapping: {df_col} → {template_col}")
            else:
                print(
                    "Invalid format. Use 'df_column_name:template_column_name' or 'remove:df_column_name'"
                )

            # Display current mapping
            print("\nCurrent mappings:")
            for df_col, template_col in mapping.items():
                print(f"- {df_col} → {template_col}")

    return mapping


def interactive_analyze_dataframe(
    df: pd.DataFrame, template: Dict[str, str], llm_provider: str = "openai"
):
    """
    Interactively analyze a DataFrame with user feedback on column mappings.

    Args:
        df: The pandas DataFrame to analyze
        template: A dictionary where keys are column names and values are expected data types
        llm_provider: The LLM provider to use for analysis

    Returns:
        A tuple containing:
            1. A dictionary mapping DataFrame column names to their analysis results
            2. A summary of the analysis results
            3. A rename dictionary that can be applied to the DataFrame
    """
    print("\nAnalyzing DataFrame...")

    # Get initial column matches
    column_matches = match_columns(df, template, llm_provider)

    # Get user feedback on mappings
    user_mapping = get_user_feedback_on_mappings(df, template, column_matches)

    # Analyze the DataFrame with user mappings
    analysis_results, summary, rename_dict = analyze_dataframe(
        df, template, llm_provider=llm_provider, rename_mapping=user_mapping
    )

    # Print the formatted results
    print(format_analysis_results(summary, rename_dict))

    # Show how to apply the rename dictionary
    print("\nTo apply the rename mapping to your DataFrame, use:")
    print("```python")
    print(f"rename_dict = {rename_dict}")
    print("df_renamed = df.rename(columns=rename_dict)")
    print("```")

    return analysis_results, summary, rename_dict
