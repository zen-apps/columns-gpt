import os
import pandas as pd
import random
import re
from typing import Dict, List, Optional, Any, Tuple, Set
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class ColumnAnalysis(BaseModel):
    """Model for structured column analysis output."""

    df_column_name: str = Field(
        description="The name of the column in the DataFrame being analyzed"
    )
    template_column_name: Optional[str] = Field(
        description="The matched column name from the template, if any"
    )
    inferred_type: str = Field(
        description="The inferred data type of the column (e.g., int, float, str, date, etc.)"
    )
    confidence: float = Field(description="Confidence score between 0 and 1")
    sample_values: List[str] = Field(
        description="Sample values from the column that were analyzed"
    )
    matches_template: Optional[bool] = Field(
        description="Whether the column matches the template requirement"
    )
    template_type: Optional[str] = Field(
        description="The expected type from the template, if provided"
    )
    match_confidence: Optional[float] = Field(
        description="Confidence score for the column name match (0-1)"
    )
    notes: Optional[str] = Field(
        description="Any additional notes or observations about the column"
    )


class ColumnMatchCandidate(BaseModel):
    """Model for column name matching candidates."""

    df_column_name: str = Field(description="The name of the column in the DataFrame")
    template_column_name: str = Field(
        description="The name of the column in the template"
    )
    match_confidence: float = Field(description="Confidence score for the match (0-1)")
    reasoning: str = Field(description="Reasoning for the match confidence")


class ColumnsMatchResponse(BaseModel):
    """Model for the column matching API response."""

    matches: List[ColumnMatchCandidate] = Field(
        description="List of potential column matches with confidence scores"
    )


class AnalysisSummary(BaseModel):
    """Model for the overall analysis summary."""

    total_df_columns: int = Field(
        description="Total number of columns in the DataFrame"
    )
    total_template_columns: int = Field(
        description="Total number of columns in the template"
    )
    analyzed_columns: int = Field(description="Number of columns that were analyzed")
    matching_columns: int = Field(
        description="Number of columns that match the template"
    )
    non_matching_columns: int = Field(
        description="Number of columns that do not match the template"
    )
    unmatched_df_columns: int = Field(
        description="Number of DataFrame columns that weren't matched to template columns"
    )
    unmatched_template_columns: int = Field(
        description="Number of template columns that weren't matched to DataFrame columns"
    )
    column_analyses: List[ColumnAnalysis] = Field(
        description="Detailed analysis for each analyzed column"
    )


def return_llm(provider="openai"):
    """Initialize and return an LLM based on the specified provider."""
    if provider == "google":
        MODEL = os.getenv("LLM_MODEL_GOOGLE", "gemini-2.0-flash-exp")  # Default model
        llm = ChatGoogleGenerativeAI(
            model=os.getenv("GOOGLE_MODEL", MODEL),
            temperature=0.0,
        )
    elif provider == "anthropic":
        MODEL = os.getenv("LLM_MODEL_ANTHROPIC", "claude-3-5-sonnet-latest")
        llm = ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", MODEL),
            temperature=0.0,
        )
    elif provider == "deepseek":
        MODEL = os.getenv("LLM_MODEL_DEEPSEEK", "deepseek-chat")  # Default model
        llm = ChatDeepSeek(
            model=MODEL,
            temperature=0.0,
        )
    elif provider == "openai":  # Default to OpenAI
        MODEL = os.getenv("LLM_MODEL_OPENAI", "gpt-4o")  # Default model
        llm = ChatOpenAI(
            model=MODEL,
            temperature=0.0,
        )
    return llm


def get_column_analysis_prompt() -> str:
    """Returns the prompt template for column type analysis."""
    return """
    You are an expert data analyst tasked with determining the data type of a column in a pandas DataFrame.
    
    Column Name: {column_name}
    
    Sample values from the column (up to 10 random samples):
    {sample_values}
    
    Please analyze the column and determine its most likely data type. Consider standard Python and pandas data types 
    (such as int, float, str, bool, datetime, timedelta, category, etc.) and provide your confidence level.
    
    If there's an expected data type from a template, it is: {expected_type}
    
    Respond with a detailed analysis of the column's data type, including:
    1. The inferred data type
    2. Your confidence level (0-1)
    3. Whether the column matches the expected type from the template
    4. Any observations about the data in the column
    
    Remember to consider edge cases like mixed types, NaN values, or special formats.
    """


def get_column_matching_prompt() -> str:
    """Returns the prompt template for matching DataFrame columns to template columns."""
    return """
    You are an expert data analyst tasked with matching columns from a pandas DataFrame to columns in a template.
    
    DataFrame columns:
    {df_columns}
    
    Template columns and their expected data types:
    {template_columns}
    
    For each DataFrame column, identify the most likely matching column in the template based on name semantics, 
    common naming patterns, and potential data types. Provide a match confidence score between 0 and 1, 
    where 1 means you're completely confident in the match.
    
    Consider the following in your matching:
    1. Semantic similarity (e.g., "first_name" and "fname" likely refer to the same concept)
    2. Common abbreviations (e.g., "id" for "identifier", "num" for "number")
    3. Word order (e.g., "customer_name" and "name_of_customer")
    4. Special characters and formatting (e.g., "user-id" and "user_id")
    5. In each match, explain why you believe they correspond
    
    Each DataFrame column might match to one template column, or it might not match any template column. 
    Each template column might be matched to multiple DataFrame columns.
    
    For each DataFrame column, list the top matching template column (if any) with a confidence score and reasoning.
    
    The DataFrame columns with sample data are:
    {df_samples}
    """


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
    matched_template_columns = set(
        match.template_column_name for match in column_matches
    )

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


def format_analysis_results(
    summary: AnalysisSummary, rename_dict: Optional[Dict[str, str]] = None
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


# Added function to get user feedback on column mappings
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


# Function to interactively analyze a DataFrame
def interactive_analyze_dataframe(
    df: pd.DataFrame, template: Dict[str, str], llm_provider: str = "openai"
):
    """
    Interactively analyze a DataFrame with user feedback on column mappings.

    Args:
        df: The pandas DataFrame to analyze
        template: A dictionary where keys are column names and values are expected data types
        llm_provider: The LLM provider to use for analysis
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


# Example usage
if __name__ == "__main__":
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

    # Option 1: Non-interactive analysis
    print("Running non-interactive analysis...")
    analysis_results, summary, rename_dict = analyze_dataframe(
        df, template, llm_provider="openai", sample_size=10, match_threshold=0.5
    )
    print(format_analysis_results(summary, rename_dict))

    # Option 2: Interactive analysis (uncomment to use)
    # print("\nRunning interactive analysis...")
    # interactive_analyze_dataframe(df, template, llm_provider="openai")
