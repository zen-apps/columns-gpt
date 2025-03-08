"""
Prompts used by ColumnsGPT for LLM interactions.
"""


def get_column_analysis_prompt() -> str:
    """Returns the prompt template for column type analysis.
    
    Returns:
        str: The prompt template string
    """
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
    """Returns the prompt template for matching DataFrame columns to template columns.
    
    Returns:
        str: The prompt template string
    """
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