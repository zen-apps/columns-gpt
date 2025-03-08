"""
Data models for GeoGPT
"""

from typing import Optional
try:
    # Try to import from pydantic directly first
    from pydantic import BaseModel, Field
except ImportError:
    # Fall back to pydantic.v1 for compatibility
    try:
        from pydantic.v1 import BaseModel, Field
    except ImportError:
        # Last resort, use langchain's import (deprecated but still works)
        from langchain_core.pydantic_v1 import BaseModel, Field


class GeoLocation(BaseModel):
    """Model for structured geographic data output."""
    
    country: str = Field(
        description="The three-letter country code (e.g., USA, CAN, GBR)"
    )
    country_full: str = Field(description="The full country name")
    postal_code: str = Field(description="The postal/zip code for the location")
    city: str = Field(description="The city name")
    state_full: Optional[str] = Field(
        description="State, province, or region depending on country"
    )
    state_code: Optional[str] = Field(
        description="Two-letter state/province, region code if applicable"
    )
    latitude: float = Field(description="The latitude coordinate in decimal degrees")
    longitude: float = Field(description="The longitude coordinate in decimal degrees")
    accuracy: str = Field(
        description="Estimated accuracy of the geocoding (high, medium, low)"
    )
    formatted_address: Optional[str] = Field(description="Complete formatted address")