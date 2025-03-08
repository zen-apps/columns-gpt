"""
ColumnsGPT: DataFrame column analysis and matching with LLM
"""

__version__ = "0.1.0"

from .models import (
    ColumnAnalysis,
    ColumnMatchCandidate,
    ColumnsMatchResponse,
    AnalysisSummary
)
from .analyzer import (
    analyze_dataframe,
    analyze_column,
    match_columns,
    interactive_analyze_dataframe
)
from .utils import (
    sample_dataframe,
    determine_type_match,
    format_analysis_results
)