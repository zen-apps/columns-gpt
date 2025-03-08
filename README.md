# GeoGPT

Enhanced geocoding with pgeocode and LLM fallback

## Overview

GeoGPT is a Python package that provides enhanced geocoding capabilities by combining the efficiency of pgeocode with the power of large language models (LLMs). The package first attempts to geocode using pgeocode for direct lookups, and if that fails or provides incomplete information, it falls back to using an LLM to fill in the gaps.

## Features

- Efficient geocoding with pgeocode
- LLM fallback for improved accuracy and coverage
- Support for multiple LLM providers (OpenAI, Google, Anthropic, DeepSeek)
- Distance calculations between locations
- Find nearby locations within a radius
- Command-line interface

## Installation

```bash
pip install geo-gpt
```

## Requirements

- Python 3.8+
- pgeocode
- pandas
- numpy
- langchain and related packages
- python-dotenv
- pycountry (optional, for enhanced country code handling)

## Setting Up Environment Variables (IMPORTANT)

GeoGPT requires environment variables to be set for LLM access. Follow these steps:

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
   export LLM_MODEL_ANTHROPIC=claude-3-7-sonnet-latest
   
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
from dotenv import load_dotenv

# Load environment variables (in development)
load_dotenv()  # Will look for .env in current directory

# Or set them directly (better for production)
os.environ["LLM_PROVIDER"] = "openai"
os.environ["OPENAI_API_KEY"] = "your-api-key-here"
os.environ["LLM_MODEL_OPENAI"] = "gpt-4o"  # Optional

# Now import and use the package
from geo_gpt import GeoCoder

# Initialize the geocoder
geocoder = GeoCoder()

# Geocode a location
location = geocoder.geocode(
    city_name="San Francisco",
    state_name="California",
    country="US"
)

# Get the results
print(f"City: {location.city}")
print(f"Coordinates: {location.latitude}, {location.longitude}")
print(f"Postal Code: {location.postal_code}")
```

## Command Line Interface

The command-line interface automatically attempts to load environment variables from a `.env` file in the current directory:

```bash
# Geocode a location
geo-gpt geocode --city "San Francisco" --state "California" --country "US" --pretty

# Calculate distance between locations
geo-gpt distance --origin "90210" --destination "10001" --country "US"

# Find nearby locations
geo-gpt nearby --reference "90210" --radius 10 --country "US" --pretty
```

## API Reference

### GeoCoder

The main class for geocoding operations.

```python
geocoder = GeoCoder(cache_dir=None, llm_provider=None)
```

**Parameters:**
- `cache_dir` (optional): Directory to store geocoding cache data
- `llm_provider` (optional): LLM provider to use for fallback ("openai", "google", "anthropic", "deepseek")

**Methods:**

- `geocode(city_name="", state_name="", zip_code="", business_name="", country="", use_llm=True) -> GeoLocation`
  
  Geocode a location using pgeocode with LLM fallback.

- `calculate_distance(origin, destination, country_code=None) -> float`
  
  Calculate the distance between two locations.

- `find_nearby_locations(reference, radius_km=50.0, country_code=None, postal_codes=None) -> List[Dict]`
  
  Find locations within a certain radius of a reference location.

### GeoLocation

A data model representing a geocoded location.

**Attributes:**
- `country`: Three-letter country code
- `country_full`: Full country name
- `postal_code`: Postal/ZIP code
- `city`: City name
- `state_full`: State/province/region name
- `state_code`: Two-letter state/province code
- `latitude`: Latitude coordinate
- `longitude`: Longitude coordinate
- `accuracy`: Estimated accuracy ("high", "medium", "low")
- `formatted_address`: Complete formatted address

## Environment Variables

These environment variables configure GeoGPT's behavior:

### Required Variables

For geocoding with LLM fallback, you MUST set:

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
   - `LLM_MODEL_ANTHROPIC`: Anthropic model (default: "claude-3-7-sonnet-latest")
   - `LLM_MODEL_DEEPSEEK`: DeepSeek model (default: "deepseek-chat")

## Troubleshooting

1. **API Key errors**: Make sure your API key is correct and has appropriate permissions

2. **ImportError for pycountry**: This optional dependency improves country code handling
   ```bash
   pip install pycountry
   ```

3. **LLM provider errors**: Check that you've set the correct environment variables for your chosen provider

4. **"No module named dotenv"**: Install python-dotenv if you're using .env files
   ```bash
   pip install python-dotenv
   ```

## License

MIT
