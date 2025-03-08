"""
Main geocoding module for GeoGPT

This module provides geocoding functionality by combining pgeocode
with LLM fallback for more complex or incomplete geocoding requests.
"""

import os
import logging
from typing import Dict, Any, Optional, List, Tuple, Union
import pandas as pd
import numpy as np
import pgeocode

from .models import GeoLocation
from .llm import get_llm
from .prompts import create_geo_prompt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class GeoCoder:
    """
    A geocoding service that combines pgeocode with LLM fallback capabilities.
    
    This class first attempts to use pgeocode for efficient direct geocoding.
    If pgeocode cannot provide complete information, it falls back to an LLM
    to enhance the results.
    """
    
    def __init__(self, cache_dir: Optional[str] = None, llm_provider: Optional[str] = None):
        """
        Initialize the GeoCoder.
        
        Args:
            cache_dir (str, optional): Directory to store geocoding cache data.
                If None, uses the package's data directory.
            llm_provider (str, optional): LLM provider to use for fallback.
                Options: "openai", "google", "anthropic", "deepseek"
                If None, uses the LLM_PROVIDER environment variable or defaults to "openai".
        """
        self.llm_provider = llm_provider
        
        # Initialize pgeocode services as needed
        self._nominatim_cache = {}
        self._geodistance_cache = {}
        
        # Initialize the LLM
        self._llm = None  # Lazy-loaded when needed
        self._geo_prompt = create_geo_prompt()
        
        # Set up cache directory
        if cache_dir is None:
            import os
            package_dir = os.path.dirname(os.path.abspath(__file__))
            self.cache_dir = os.path.join(package_dir, "data")
        else:
            self.cache_dir = cache_dir
            
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def _get_nominatim(self, country_code: str) -> pgeocode.Nominatim:
        """
        Get or create a pgeocode Nominatim instance for a country.
        
        Args:
            country_code (str): The two-letter country code
            
        Returns:
            pgeocode.Nominatim: The Nominatim instance for the country
        """
        if country_code not in self._nominatim_cache:
            self._nominatim_cache[country_code] = pgeocode.Nominatim(country_code)
        return self._nominatim_cache[country_code]
        
    def _get_geodistance(self, country_code: str) -> pgeocode.GeoDistance:
        """
        Get or create a pgeocode GeoDistance instance for a country.
        
        Args:
            country_code (str): The two-letter country code
            
        Returns:
            pgeocode.GeoDistance: The GeoDistance instance for the country
        """
        if country_code not in self._geodistance_cache:
            self._geodistance_cache[country_code] = pgeocode.GeoDistance(country_code)
        return self._geodistance_cache[country_code]
    
    def _get_llm(self):
        """
        Get or initialize the LLM for fallback geocoding.
        
        Returns:
            LLM: The LangChain LLM instance
        """
        if self._llm is None:
            self._llm = get_llm(self.llm_provider)
        return self._llm
        
    def _pgeocode_to_geolocation(self, result: pd.Series) -> Optional[GeoLocation]:
        """
        Convert pgeocode result to GeoLocation model.
        
        Args:
            result (pd.Series): pgeocode result
            
        Returns:
            Optional[GeoLocation]: The converted GeoLocation, or None if invalid
        """
        if result is None or result.empty or pd.isna(result.iloc[0]):
            return None
            
        try:
            # Determine country code (pgeocode uses 2-letter codes)
            country_code = str(result.get('country_code', '')).strip()
            if not country_code:
                return None
                
            # Map 2-letter to 3-letter country code
            import pycountry
            try:
                country = pycountry.countries.get(alpha_2=country_code)
                country_3letter = country.alpha_3 if country else country_code
                country_full = country.name if country else 'Unknown'
            except (AttributeError, ImportError):
                # Fallback if pycountry not available
                country_3letter = country_code
                country_full = 'Unknown'
            
            # Extract state/province info
            state_full = str(result.get('state_name', '')) if pd.notna(result.get('state_name', None)) else None
            state_code = str(result.get('state_code', '')) if pd.notna(result.get('state_code', None)) else None
            
            # Get place name and postal code
            place_name = str(result.get('place_name', '')) if pd.notna(result.get('place_name', None)) else None
            postal_code = str(result.get('postal_code', '')) if pd.notna(result.get('postal_code', None)) else None
            
            # Extract city from place name if possible
            city = place_name.split(',')[0] if place_name else None
            
            # Get coordinates
            latitude = float(result.get('latitude')) if pd.notna(result.get('latitude', None)) else None
            longitude = float(result.get('longitude')) if pd.notna(result.get('longitude', None)) else None
            
            # Construct formatted address
            address_parts = []
            if place_name:
                address_parts.append(place_name)
            if state_full:
                address_parts.append(state_full)
            if country_full and country_full != 'Unknown':
                address_parts.append(country_full)
                
            formatted_address = ", ".join(address_parts) if address_parts else None
            
            # Determine accuracy based on available details
            if latitude is not None and longitude is not None and postal_code:
                accuracy = "high"
            elif (latitude is not None and longitude is not None) or postal_code:
                accuracy = "medium"
            else:
                accuracy = "low"
                
            return GeoLocation(
                country=country_3letter,
                country_full=country_full,
                postal_code=postal_code or "",
                city=city or "",
                state_full=state_full,
                state_code=state_code,
                latitude=latitude or 0.0,
                longitude=longitude or 0.0,
                accuracy=accuracy,
                formatted_address=formatted_address
            )
            
        except Exception as e:
            logger.error(f"Error converting pgeocode result to GeoLocation: {e}")
            return None
    
    def _geocode_with_pgeocode(
        self, 
        zip_code: Optional[str] = None,
        city_name: Optional[str] = None,
        state_name: Optional[str] = None, 
        country_code: Optional[str] = None
    ) -> Optional[GeoLocation]:
        """
        Attempt geocoding using pgeocode.
        
        Args:
            zip_code (str, optional): Postal/ZIP code
            city_name (str, optional): City name
            state_name (str, optional): State/province name
            country_code (str, optional): Two-letter country code
            
        Returns:
            Optional[GeoLocation]: The geocoded location or None if unsuccessful
        """
        # Can't do anything without a country code
        if not country_code:
            return None
            
        try:
            # Try direct postal code lookup if available
            if zip_code:
                nomi = self._get_nominatim(country_code)
                result = nomi.query_postal_code(zip_code)
                location = self._pgeocode_to_geolocation(result)
                if location and location.city and location.latitude != 0.0:
                    return location
                    
            # Try city name lookup if available
            if city_name:
                nomi = self._get_nominatim(country_code)
                results = nomi.query_location(city_name, fuzzy_threshold=80)
                
                # Filter by state if provided
                if state_name and not results.empty:
                    # Convert state_name column to string first to avoid 'float has no attribute lower' error
                    results['state_name'] = results['state_name'].astype(str)
                    state_filtered = results[
                        results['state_name'].str.contains(state_name, case=False, na=False)
                    ]
                    if not state_filtered.empty:
                        results = state_filtered
                
                if not results.empty:
                    location = self._pgeocode_to_geolocation(results.iloc[0])
                    if location:
                        return location
                        
            return None
            
        except Exception as e:
            logger.error(f"Error in pgeocode geocoding: {e}")
            return None
    
    def _geocode_with_llm(
        self,
        city_name: str = "",
        state_name: str = "",
        zip_code: str = "",
        business_name: str = "",
        country: str = "",
    ) -> GeoLocation:
        """
        Geocode using an LLM.
        
        Args:
            city_name (str): City name
            state_name (str): State/province name
            zip_code (str): Postal/ZIP code
            business_name (str): Business name at location
            country (str): Country name or code
            
        Returns:
            GeoLocation: The geocoded location
        """
        # Get the LLM
        llm = self._get_llm()
        
        # Set up LLM for structured output
        llm_with_structured_output = llm.with_structured_output(GeoLocation)
        
        # Fill in the prompt template
        prompt_filled_in = self._geo_prompt.format(
            country=country,
            city_name=city_name,
            state_name=state_name,
            zip_code=zip_code,
            business_name=business_name,
        )
        
        # Get the result from the LLM
        try:
            result = llm_with_structured_output.invoke(prompt_filled_in)
            
            # Modify the formatted address to indicate LLM usage
            original_address = result.formatted_address
            if original_address:
                result.formatted_address = f"{original_address} [LLM]"
            else:
                result.formatted_address = "[LLM]"
                
            return result
        except Exception as e:
            logger.error(f"Error in LLM geocoding: {e}")
            # Provide a fallback result
            return GeoLocation(
                country="UNK",
                country_full="Unknown",
                postal_code="",
                city="",
                state_full=None,
                state_code=None,
                latitude=0.0,
                longitude=0.0,
                accuracy="low",
                formatted_address="[LLM Error]"
            )
    
    def _iso2_to_iso3_country(self, iso2_code: str) -> str:
        """
        Convert ISO 2-letter country code to 3-letter code.
        
        Args:
            iso2_code (str): The 2-letter country code
            
        Returns:
            str: The 3-letter country code or original if conversion fails
        """
        try:
            import pycountry
            country = pycountry.countries.get(alpha_2=iso2_code.upper())
            return country.alpha_3 if country else iso2_code
        except (ImportError, AttributeError):
            # Simple mapping for common countries if pycountry not available
            mapping = {
                'US': 'USA', 'CA': 'CAN', 'GB': 'GBR', 'FR': 'FRA', 'DE': 'DEU',
                'JP': 'JPN', 'CN': 'CHN', 'AU': 'AUS', 'NZ': 'NZL', 'RU': 'RUS'
            }
            return mapping.get(iso2_code.upper(), iso2_code)
    
    def _iso3_to_iso2_country(self, iso3_code: str) -> str:
        """
        Convert ISO 3-letter country code to 2-letter code.
        
        Args:
            iso3_code (str): The 3-letter country code
            
        Returns:
            str: The 2-letter country code or original if conversion fails
        """
        try:
            import pycountry
            country = pycountry.countries.get(alpha_3=iso3_code.upper())
            return country.alpha_2 if country else iso3_code
        except (ImportError, AttributeError):
            # Simple mapping for common countries if pycountry not available
            mapping = {
                'USA': 'US', 'CAN': 'CA', 'GBR': 'GB', 'FRA': 'FR', 'DEU': 'DE',
                'JPN': 'JP', 'CHN': 'CN', 'AUS': 'AU', 'NZL': 'NZ', 'RUS': 'RU'
            }
            return mapping.get(iso3_code.upper(), iso3_code[:2])
            
    def _normalize_country_code(self, country: str) -> Optional[str]:
        """
        Normalize a country name or code to a 2-letter ISO code.
        
        Args:
            country (str): Country name or code
            
        Returns:
            Optional[str]: Normalized 2-letter country code or None if not found
        """
        if not country:
            return None
            
        # If it's already a 2-letter code
        if len(country) == 2:
            return country.upper()
            
        # If it's a 3-letter code
        if len(country) == 3:
            return self._iso3_to_iso2_country(country)
            
        # Try to lookup by name
        try:
            import pycountry
            results = pycountry.countries.search_fuzzy(country)
            if results:
                return results[0].alpha_2
            return None
        except (ImportError, LookupError):
            # Fallback for common countries if pycountry not available
            common_countries = {
                'united states': 'US', 'usa': 'US', 'america': 'US',
                'canada': 'CA', 'united kingdom': 'GB', 'uk': 'GB',
                'france': 'FR', 'germany': 'DE', 'japan': 'JP',
                'china': 'CN', 'australia': 'AU', 'new zealand': 'NZ',
                'russia': 'RU', 'mexico': 'MX', 'brazil': 'BR',
                'india': 'IN', 'spain': 'ES', 'italy': 'IT'
            }
            return common_countries.get(country.lower(), None)
    
    def geocode(
        self,
        city_name: str = "",
        state_name: str = "",
        zip_code: str = "",
        business_name: str = "",
        country: str = "",
        use_llm: bool = True
    ) -> GeoLocation:
        """
        Geocode a location using pgeocode with LLM fallback.
        
        This method first attempts to use pgeocode for efficient geocoding.
        If pgeocode cannot provide complete information, it falls back to
        an LLM to enhance the results.
        
        Args:
            city_name (str, optional): City name
            state_name (str, optional): State/province name
            zip_code (str, optional): Postal/ZIP code
            business_name (str, optional): Business name at location
            country (str, optional): Country name or code
            use_llm (bool, optional): Whether to use LLM fallback if needed
            
        Returns:
            GeoLocation: The geocoded location
        """
        # First, try to normalize the country code
        country_code = self._normalize_country_code(country)
        
        # Try to geocode with pgeocode first
        if country_code:
            pg_result = self._geocode_with_pgeocode(
                zip_code=zip_code,
                city_name=city_name,
                state_name=state_name,
                country_code=country_code
            )
            
            # If pgeocode was successful, return the result
            if pg_result and pg_result.accuracy != "low":
                logger.info(f"Successfully geocoded with pgeocode: {pg_result.city}, {pg_result.country}")
                return pg_result
        
        # If pgeocode failed or wasn't enough, and LLM is allowed, try LLM
        if use_llm:
            logger.info("Falling back to LLM for geocoding")
            llm_result = self._geocode_with_llm(
                city_name=city_name,
                state_name=state_name,
                zip_code=zip_code,
                business_name=business_name,
                country=country,
            )
            return llm_result
            
        # If LLM not allowed and pgeocode failed, return the limited pgeocode result
        if pg_result:
            return pg_result
            
        # Last resort - return empty result
        return GeoLocation(
            country="UNK",
            country_full="Unknown",
            postal_code="",
            city="",
            state_full=None,
            state_code=None,
            latitude=0.0,
            longitude=0.0,
            accuracy="low",
            formatted_address=None
        )
    
    def calculate_distance(
        self,
        origin: Union[str, GeoLocation],
        destination: Union[str, GeoLocation],
        country_code: Optional[str] = None
    ) -> float:
        """
        Calculate the distance between two locations.
        
        Args:
            origin (Union[str, GeoLocation]): Origin postal code or GeoLocation
            destination (Union[str, GeoLocation]): Destination postal code or GeoLocation
            country_code (str, optional): Country code for the postal codes
            
        Returns:
            float: Distance in kilometers
        """
        # Extract postal codes if GeoLocation objects were provided
        origin_code = origin.postal_code if isinstance(origin, GeoLocation) else origin
        dest_code = destination.postal_code if isinstance(destination, GeoLocation) else destination
        
        # If no country code provided, try to get it from GeoLocation
        if country_code is None:
            if isinstance(origin, GeoLocation):
                country_code = self._iso3_to_iso2_country(origin.country)
            elif isinstance(destination, GeoLocation):
                country_code = self._iso3_to_iso2_country(destination.country)
                
        # Can't proceed without a country code
        if not country_code:
            raise ValueError("Country code must be provided for postal code distance calculation")
            
        try:
            # Calculate distance using pgeocode
            dist = self._get_geodistance(country_code)
            distance = dist.query_postal_code(origin_code, dest_code)
            return float(distance)
        except Exception as e:
            logger.error(f"Error calculating distance: {e}")
            
            # Fallback to haversine formula if coordinates are available
            if isinstance(origin, GeoLocation) and isinstance(destination, GeoLocation):
                if origin.latitude != 0 and origin.longitude != 0 and \
                   destination.latitude != 0 and destination.longitude != 0:
                    return self._haversine_distance(
                        origin.latitude, origin.longitude,
                        destination.latitude, destination.longitude
                    )
            
            # Return -1 if calculation failed
            return -1.0
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate the great circle distance between two points on the earth.
        
        Args:
            lat1 (float): Origin latitude
            lon1 (float): Origin longitude
            lat2 (float): Destination latitude
            lon2 (float): Destination longitude
            
        Returns:
            float: Distance in kilometers
        """
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371  # Radius of earth in kilometers
        return c * r
    
    def find_nearby_locations(
        self, 
        reference: Union[str, GeoLocation],
        radius_km: float = 50.0,
        country_code: Optional[str] = None,
        postal_codes: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Find locations within a certain radius of a reference location.
        
        Args:
            reference (Union[str, GeoLocation]): Reference postal code or GeoLocation
            radius_km (float): Search radius in kilometers
            country_code (str, optional): Country code for the postal codes
            postal_codes (List[str], optional): List of postal codes to check
            
        Returns:
            List[Dict[str, Any]]: Nearby locations with distance information
        """
        # Extract postal code if GeoLocation was provided
        ref_code = reference.postal_code if isinstance(reference, GeoLocation) else reference
        
        # If no country code provided, try to get it from GeoLocation
        if country_code is None and isinstance(reference, GeoLocation):
            country_code = self._iso3_to_iso2_country(reference.country)
                
        # Can't proceed without a country code
        if not country_code:
            raise ValueError("Country code must be provided for nearby location search")
            
        try:
            # Get the distance calculator
            dist = self._get_geodistance(country_code)
            nomi = self._get_nominatim(country_code)
            
            # If no postal codes provided, use a limited set from the API
            if postal_codes is None:
                # For US, try to load from our dataset
                if country_code.upper() == 'US':
                    try:
                        import os
                        csv_path = os.path.join(self.cache_dir, 'us_zips.csv')
                        # If we don't have the file, copy it from package data
                        if not os.path.exists(csv_path):
                            import shutil
                            import pkg_resources
                            data_path = pkg_resources.resource_filename('geo_gpt', 'data/us_zips.csv')
                            shutil.copy(data_path, csv_path)
                            
                        zips_df = pd.read_csv(csv_path)
                        postal_codes = zips_df['zip'].astype(str).tolist()
                    except Exception as e:
                        logger.error(f"Error loading US zip codes: {e}")
                        # Fallback to a small sample for testing
                        postal_codes = [f"{i:05d}" for i in range(10001, 10201)]
                else:
                    # For demo purposes only - in real app, would load from a comprehensive dataset
                    postal_codes = [f"{i:05d}" for i in range(10000, 10100)]
            
            # Calculate distances to all codes
            try:
                distances = dist.query_postal_code(ref_code, postal_codes)
            except Exception as e:
                logger.error(f"Error calculating distances: {e}")
                return []
            
            # Filter by radius
            nearby_indices = np.where(distances <= radius_km)[0]
            nearby_codes = [postal_codes[i] for i in nearby_indices]
            
            # Get details for nearby postal codes
            if nearby_codes:
                try:
                    nearby_data = nomi.query_postal_code(nearby_codes)
                    
                    # Convert to list of dicts with distance
                    result = []
                    for i, code in enumerate(nearby_codes):
                        idx = nearby_indices[i]
                        row_data = nearby_data.iloc[i].to_dict() if i < len(nearby_data) else {}
                        
                        # Add distance
                        row_data['distance_km'] = float(distances[idx])
                        
                        # Add postal code if missing
                        if 'postal_code' not in row_data or pd.isna(row_data['postal_code']):
                            row_data['postal_code'] = code
                            
                        result.append(row_data)
                    
                    # Sort by distance
                    return sorted(result, key=lambda x: x['distance_km'])
                except Exception as e:
                    logger.error(f"Error processing nearby postal codes: {e}")
                    return []
            
            return []
            
        except Exception as e:
            logger.error(f"Error finding nearby locations: {e}")
            return []