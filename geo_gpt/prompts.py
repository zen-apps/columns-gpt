"""
Prompt templates for GeoGPT
"""

from langchain.prompts import PromptTemplate


def get_geo_prompt() -> str:
    """
    Returns the prompt template for geocoding with an LLM.
    """
    return """You are a highly accurate geocoding assistant specializing in global locations. Your task is to provide complete geographic information for the location described.

LOCATION DETAILS PROVIDED:
- Country: {country}
- City: {city_name}
- State/Province/Region: {state_name}
- Postal/ZIP Code: {zip_code}
- Business Name (if applicable): {business_name}

OUTPUT REQUIREMENTS:
1. Provide all geographic details in the structured output format.
2. For country code, use the official 3-letter ISO code (e.g., USA for United States).
3. Include the full country name in country_full.
4. If coordinates (latitude/longitude) are known, provide them to 6 decimal places.
5. For accuracy, indicate "high" if you're certain, "medium" if reasonably confident, or "low" if uncertain.
6. Only return None for fields where information cannot not be determined.
7. For US states, include the 2-letter state code (e.g., MN for Minnesota).
8. Provide a properly formatted complete address in formatted_address.
9. If the city is not known, use the postal code and other information to infer it.
10. Do not provide latitude/longitude if the postal code is not known.

Think step-by-step:
1. First, identify the country based on the provided information.
2. Then, determine each geographic component using your knowledge.
3. For coordinates, use known geographic positions of the city or postal code.
4. Verify consistency between city, state, and postal code.
5. Format the address according to local conventions.

Remember that accuracy is critical for geocoding applications."""


def create_geo_prompt() -> PromptTemplate:
    """
    Creates a LangChain PromptTemplate for geocoding.
    
    Returns:
        PromptTemplate: The geocoding prompt template
    """
    return PromptTemplate(
        template=get_geo_prompt(),
        input_variables=[
            "country",
            "city_name",
            "state_name",
            "zip_code",
            "business_name",
        ],
    )