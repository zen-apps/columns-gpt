"""
Command Line Interface for GeoGPT
"""

import os
import sys
import argparse
import json
import logging
from typing import Dict, Any, Optional, List

from dotenv import load_dotenv
from .geocoder import GeoCoder, GeoLocation

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_environment_setup() -> None:
    """
    Check if required environment variables are set and provide helpful messages
    if they're missing.
    """
    # Try to load environment variables from .env file
    try:
        load_dotenv()
        logger.info("Loaded environment variables from .env file")
    except Exception as e:
        logger.warning(f"Could not load .env file: {e}")

    # Check which provider is set
    provider = os.getenv("LLM_PROVIDER")
    if not provider:
        logger.warning(
            "LLM_PROVIDER environment variable not set. Defaulting to openai."
        )
        provider = "openai"
        os.environ["LLM_PROVIDER"] = provider

    logger.info(f"Using LLM provider: {provider}")

    # Check for API key based on provider
    api_key_var = None
    if provider == "openai":
        api_key_var = "OPENAI_API_KEY"
    elif provider == "google":
        api_key_var = "GOOGLE_API_KEY"
    elif provider == "anthropic":
        api_key_var = "ANTHROPIC_API_KEY"
    elif provider == "deepseek":
        api_key_var = "DEEPSEEK_API_KEY"

    if api_key_var and not os.getenv(api_key_var):
        logger.warning(f"⚠️  {api_key_var} environment variable not set!")
        logger.warning(f"LLM fallback will not work without a valid API key.")
        logger.warning(f"Set it with: export {api_key_var}=your-api-key-here")
        logger.warning(
            "See README.md for more information on setting up environment variables."
        )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="GeoGPT - Enhanced geocoding with pgeocode and LLM fallback",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Add environment variable setup argument
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Check environment variables setup and print status",
    )

    # Create subparsers
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Geocode command
    geocode_parser = subparsers.add_parser("geocode", help="Geocode a location")
    geocode_parser.add_argument("--city", help="City name")
    geocode_parser.add_argument("--state", help="State/province name")
    geocode_parser.add_argument("--zip", help="Postal/ZIP code")
    geocode_parser.add_argument("--business", help="Business name")
    geocode_parser.add_argument("--country", help="Country name or code")
    geocode_parser.add_argument(
        "--provider", help="LLM provider: openai, google, anthropic, deepseek"
    )
    geocode_parser.add_argument(
        "--no-llm", action="store_true", help="Disable LLM fallback, use only pgeocode"
    )
    geocode_parser.add_argument(
        "--pretty", action="store_true", help="Pretty print the output"
    )

    # Distance command
    distance_parser = subparsers.add_parser(
        "distance", help="Calculate distance between two locations"
    )
    distance_parser.add_argument("--origin", required=True, help="Origin postal code")
    distance_parser.add_argument(
        "--destination", required=True, help="Destination postal code"
    )
    distance_parser.add_argument("--country", required=True, help="Country code")

    # Nearby command
    nearby_parser = subparsers.add_parser(
        "nearby", help="Find locations nearby a reference location"
    )
    nearby_parser.add_argument(
        "--reference", required=True, help="Reference postal code"
    )
    nearby_parser.add_argument(
        "--radius", type=float, default=50.0, help="Search radius in km"
    )
    nearby_parser.add_argument("--country", required=True, help="Country code")
    nearby_parser.add_argument(
        "--limit", type=int, default=10, help="Maximum number of results"
    )
    nearby_parser.add_argument(
        "--pretty", action="store_true", help="Pretty print the output"
    )

    return parser.parse_args()


def print_environment_status() -> None:
    """Print environment variable setup status in a user-friendly way"""
    print("\nEnvironment Variables Setup:")
    print("=" * 60)

    # Check which provider is set
    provider = os.getenv("LLM_PROVIDER")
    if not provider:
        print("❌ LLM_PROVIDER: Not set (will default to 'openai')")
    else:
        print(f"✅ LLM_PROVIDER: {provider}")

    # Check provider-specific variables
    if provider == "openai" or not provider:
        api_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("LLM_MODEL_OPENAI", "gpt-4o")
        print(f"✅ LLM_MODEL_OPENAI: {model}")
        if api_key:
            print(
                f"✅ OPENAI_API_KEY: {api_key[:4]}...{api_key[-4:] if len(api_key) > 8 else ''}"
            )
        else:
            print("❌ OPENAI_API_KEY: Not set (required for LLM fallback)")

    elif provider == "google":
        api_key = os.getenv("GOOGLE_API_KEY")
        model = os.getenv("LLM_MODEL_GOOGLE", "gemini-1.5-pro")
        print(f"✅ LLM_MODEL_GOOGLE: {model}")
        if api_key:
            print(
                f"✅ GOOGLE_API_KEY: {api_key[:4]}...{api_key[-4:] if len(api_key) > 8 else ''}"
            )
        else:
            print("❌ GOOGLE_API_KEY: Not set (required for LLM fallback)")

    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        model = os.getenv("LLM_MODEL_ANTHROPIC", "claude-3-5-sonnet-latest")
        print(f"✅ LLM_MODEL_ANTHROPIC: {model}")
        if api_key:
            print(
                f"✅ ANTHROPIC_API_KEY: {api_key[:4]}...{api_key[-4:] if len(api_key) > 8 else ''}"
            )
        else:
            print("❌ ANTHROPIC_API_KEY: Not set (required for LLM fallback)")

    elif provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        model = os.getenv("LLM_MODEL_DEEPSEEK", "deepseek-chat")
        print(f"✅ LLM_MODEL_DEEPSEEK: {model}")
        if api_key:
            print(
                f"✅ DEEPSEEK_API_KEY: {api_key[:4]}...{api_key[-4:] if len(api_key) > 8 else ''}"
            )
        else:
            print("❌ DEEPSEEK_API_KEY: Not set (required for LLM fallback)")

    print("\nHow to set up environment variables:")
    print("-" * 60)
    print("Option 1: Create a .env file in the current directory:")
    print(f"  LLM_PROVIDER={provider or 'openai'}")
    if provider == "openai" or not provider:
        print("  OPENAI_API_KEY=your-api-key-here")
    elif provider == "google":
        print("  GOOGLE_API_KEY=your-api-key-here")
    elif provider == "anthropic":
        print("  ANTHROPIC_API_KEY=your-api-key-here")
    elif provider == "deepseek":
        print("  DEEPSEEK_API_KEY=your-api-key-here")

    print("\nOption 2: Set environment variables directly:")
    if provider == "openai" or not provider:
        print("  export LLM_PROVIDER=openai")
        print("  export OPENAI_API_KEY=your-api-key-here")
    elif provider == "google":
        print("  export LLM_PROVIDER=google")
        print("  export GOOGLE_API_KEY=your-api-key-here")
    elif provider == "anthropic":
        print("  export LLM_PROVIDER=anthropic")
        print("  export ANTHROPIC_API_KEY=your-api-key-here")
    elif provider == "deepseek":
        print("  export LLM_PROVIDER=deepseek")
        print("  export DEEPSEEK_API_KEY=your-api-key-here")

    print("\nFor more information, see the README.md file.")
    print("=" * 60)


def print_geo_location(location: GeoLocation, pretty: bool = False) -> None:
    """Print a GeoLocation to the console.

    Args:
        location (GeoLocation): The location to print
        pretty (bool, optional): Whether to pretty-print the output
    """
    location_dict = location.dict()

    if pretty:
        print("\nLocation Results:")
        print("-" * 40)
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
        print("-" * 40)
    else:
        print(json.dumps(location_dict))


def print_nearby_locations(
    locations: List[Dict[str, Any]], limit: int = 10, pretty: bool = False
) -> None:
    """Print nearby locations to the console.

    Args:
        locations (List[Dict[str, Any]]): The locations to print
        limit (int, optional): Maximum number of locations to print
        pretty (bool): Whether to pretty-print the output
    """
    if not locations:
        print("No locations found")
        return

    # Limit the number of results
    locations = locations[:limit]

    if pretty:
        print("\nNearby Locations:")
        print("-" * 60)
        print(f"{'Postal Code':<12} {'Place':<30} {'Distance (km)':<15}")
        print("-" * 60)

        for loc in locations:
            postal_code = loc.get("postal_code", "N/A")
            place_name = loc.get("place_name", "Unknown")
            distance = loc.get("distance_km", 0.0)

            print(f"{postal_code:<12} {place_name:<30} {distance:<15.2f}")

        print("-" * 60)
    else:
        print(json.dumps(locations))


def main() -> None:
    """Main entry point for the CLI"""
    # Check environment variables
    check_environment_setup()

    # Parse arguments
    args = parse_args()

    # If --check-env flag is provided, print status and exit
    if getattr(args, "check_env", False):
        print_environment_status()
        return

    # Handle no command
    if not getattr(args, "command", None):
        print("Please specify a command. Use --help for more information.")
        sys.exit(1)

    # Initialize the geocoder
    geocoder = GeoCoder(llm_provider=getattr(args, "provider", None))

    if args.command == "geocode":
        # Perform geocoding
        result = geocoder.geocode(
            city_name=args.city or "",
            state_name=args.state or "",
            zip_code=args.zip or "",
            business_name=args.business or "",
            country=args.country or "",
            use_llm=not args.no_llm,
        )
        print_geo_location(result, args.pretty)

    elif args.command == "distance":
        # Calculate distance
        distance = geocoder.calculate_distance(
            origin=args.origin, destination=args.destination, country_code=args.country
        )
        print(f"Distance: {distance:.2f} km")

    elif args.command == "nearby":
        # Find nearby locations
        locations = geocoder.find_nearby_locations(
            reference=args.reference,
            radius_km=args.radius,
            country_code=args.country,
        )
        print_nearby_locations(locations, args.limit, args.pretty)


if __name__ == "__main__":
    main()
