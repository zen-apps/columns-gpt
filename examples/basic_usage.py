"""
Basic usage example for GeoGPT

This example demonstrates how to use the GeoGPT package for various geocoding tasks.
Before running this example, make sure to set up environment variables for your LLM provider.
"""

import os
import logging
from dotenv import load_dotenv
from geo_gpt import GeoCoder, GeoLocation

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_environment():
    """
    Set up environment variables for LLM usage.
    In production code, you would set these variables in your environment
    rather than in code.
    """
    # Try to load from .env file first
    load_dotenv()

    # Check if we have the required environment variables
    provider = os.getenv("LLM_PROVIDER")
    if not provider:
        # Default to OpenAI if not specified
        provider = "openai"
        os.environ["LLM_PROVIDER"] = provider
        logger.info(f"Setting default LLM provider: {provider}")

    # Set up model based on provider
    if provider == "openai":
        if not os.getenv("LLM_MODEL_OPENAI"):
            os.environ["LLM_MODEL_OPENAI"] = "gpt-4o"
            logger.info("Setting default OpenAI model: gpt-4o")

        if not os.getenv("OPENAI_API_KEY"):
            logger.warning("OPENAI_API_KEY not set - geocoding with LLM will fail")
            logger.warning(
                "Set your OpenAI API key with: export OPENAI_API_KEY=your-key-here"
            )

    elif provider == "google":
        if not os.getenv("LLM_MODEL_GOOGLE"):
            os.environ["LLM_MODEL_GOOGLE"] = "gemini-2.0-flash-exp"
            logger.info("Setting default Google model: gemini-2.0-flash-exp")

        if not os.getenv("GOOGLE_API_KEY"):
            logger.warning("GOOGLE_API_KEY not set - geocoding with LLM will fail")
            logger.warning(
                "Set your Google API key with: export GOOGLE_API_KEY=your-key-here"
            )

    elif provider == "anthropic":
        if not os.getenv("LLM_MODEL_ANTHROPIC"):
            os.environ["LLM_MODEL_ANTHROPIC"] = "claude-3-7-sonnet-latest"
            logger.info("Setting default Anthropic model: claude-3-7-sonnet-latest")

        if not os.getenv("ANTHROPIC_API_KEY"):
            logger.warning("ANTHROPIC_API_KEY not set - geocoding with LLM will fail")
            logger.warning(
                "Set your Anthropic API key with: export ANTHROPIC_API_KEY=your-key-here"
            )

    elif provider == "deepseek":
        if not os.getenv("LLM_MODEL_DEEPSEEK"):
            os.environ["LLM_MODEL_DEEPSEEK"] = "deepseek-chat"
            logger.info("Setting default DeepSeek model: deepseek-chat")

        if not os.getenv("DEEPSEEK_API_KEY"):
            logger.warning("DEEPSEEK_API_KEY not set - geocoding with LLM will fail")
            logger.warning(
                "Set your DeepSeek API key with: export DEEPSEEK_API_KEY=your-key-here"
            )

    logger.info(
        f"Environment setup complete. Using provider: {os.getenv('LLM_PROVIDER')}"
    )


def main():
    # Set up environment variables before creating the geocoder
    setup_environment()

    # Initialize the geocoder after environment setup
    logger.info("Initializing geocoder")
    geocoder = GeoCoder()

    # Example 1: Geocode with ZIP code
    print("Example 1: Geocoding with ZIP code")
    location = geocoder.geocode(zip_code="90210", country="US")
    print_location(location)

    # Example 2: Geocode with city and state
    print("\nExample 2: Geocoding with city and state")
    location = geocoder.geocode(
        city_name="San Francisco", state_name="California", country="US"
    )
    print_location(location)

    # Example 3: Geocode with business name
    print("\nExample 3: Geocoding with business name (uses LLM)")
    location = geocoder.geocode(
        business_name="Carlson Craft", state_name="Minnesota", country="US"
    )
    print_location(location)

    # Example 4: Calculate distance between locations
    print("\nExample 4: Calculate distance between locations")
    distance = geocoder.calculate_distance(
        origin="90210", destination="10001", country_code="US"
    )
    print(
        f"Distance between Beverly Hills (90210) and New York (10001): {distance:.2f} km"
    )

    # Example 5: Find nearby locations
    print("\nExample 5: Find nearby locations")
    nearby = geocoder.find_nearby_locations(
        reference="90210", radius_km=10.0, country_code="US"
    )
    print(f"Found {len(nearby)} locations within 10km of Beverly Hills (90210)")

    # Print the first 3 nearby locations
    for i, loc in enumerate(nearby[:3]):
        postal_code = loc.get("postal_code", "N/A")
        place_name = loc.get("place_name", "Unknown")
        distance = loc.get("distance_km", 0.0)
        print(f"{i+1}. {postal_code}: {place_name} - {distance:.2f} km")


def print_location(location: GeoLocation):
    """Print a location in a readable format"""
    print(f"Country: {location.country} ({location.country_full})")
    print(f"City: {location.city}")

    if location.state_full:
        state_info = f"{location.state_full}"
        if location.state_code:
            state_info += f" ({location.state_code})"
        print(f"State/Province: {state_info}")

    print(f"Postal Code: {location.postal_code}")
    print(f"Coordinates: {location.latitude}, {location.longitude}")
    print(f"Accuracy: {location.accuracy}")

    if location.formatted_address:
        print(f"Address: {location.formatted_address}")


if __name__ == "__main__":
    main()
