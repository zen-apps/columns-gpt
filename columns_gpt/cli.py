"""
Command-line interface for ColumnsGPT.
"""

import argparse
import json
import os
import pandas as pd
import sys
from typing import Dict

from .analyzer import analyze_dataframe, interactive_analyze_dataframe
from .utils import format_analysis_results


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="ColumnsGPT: DataFrame column analysis and matching with LLM")
    parser.add_argument("--input", "-i", required=True, help="Input CSV or Excel file path")
    parser.add_argument(
        "--template", "-t", required=True, help="Template JSON file with column name and type mapping"
    )
    parser.add_argument(
        "--interactive", "-ia", action="store_true", help="Run in interactive mode to adjust column mappings"
    )
    parser.add_argument(
        "--provider", "-p", default="openai", choices=["openai", "google", "anthropic", "deepseek"],
        help="LLM provider to use (default: openai)"
    )
    parser.add_argument(
        "--threshold", "-th", type=float, default=0.7,
        help="Confidence threshold for column matching (default: 0.7)"
    )
    parser.add_argument(
        "--output", "-o", help="Output JSON file path for results (optional)"
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Check if template file exists
    if not os.path.exists(args.template):
        print(f"Error: Template file not found: {args.template}")
        sys.exit(1)
    
    # Load the DataFrame
    file_ext = os.path.splitext(args.input)[1].lower()
    if file_ext == ".csv":
        df = pd.read_csv(args.input)
    elif file_ext in [".xls", ".xlsx"]:
        df = pd.read_excel(args.input)
    else:
        print(f"Error: Unsupported file format: {file_ext}. Only CSV and Excel files are supported.")
        sys.exit(1)
    
    # Load the template
    with open(args.template, 'r') as f:
        template = json.load(f)
    
    # Check if template is a proper dictionary
    if not isinstance(template, dict):
        print("Error: Template must be a JSON object with column names as keys and expected types as values.")
        sys.exit(1)
    
    # Run analysis
    if args.interactive:
        _, summary, rename_dict = interactive_analyze_dataframe(
            df, template, llm_provider=args.provider
        )
    else:
        _, summary, rename_dict = analyze_dataframe(
            df, template, llm_provider=args.provider, match_threshold=args.threshold
        )
        # Print formatted results in non-interactive mode
        print(format_analysis_results(summary, rename_dict))
    
    # Save results if output file is specified
    if args.output:
        # Convert summary to dictionary for JSON serialization
        summary_dict = summary.dict()
        
        output_data = {
            "summary": summary_dict,
            "rename_mapping": rename_dict
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Results saved to: {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())