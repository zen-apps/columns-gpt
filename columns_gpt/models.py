"""
Data models for ColumnsGPT
"""

from typing import Dict, List, Optional, Any, Tuple, Set
try:
    # Try to import from pydantic directly first
    from pydantic import BaseModel, Field
except ImportError:
    # Fall back to pydantic.v1 for compatibility
    try:
        from pydantic.v1 import BaseModel, Field
    except ImportError:
        # Last resort, use langchain's import
        from langchain_core.pydantic_v1 import BaseModel, Field


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