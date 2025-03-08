"""
Tests for the GeoCoder module
"""

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd

from geo_gpt.geocoder import GeoCoder
from geo_gpt.models import GeoLocation


class TestGeoCoder(unittest.TestCase):
    """Test cases for the GeoCoder class"""
    
    def setUp(self):
        """Set up the test environment"""
        # Patch the LLM for testing
        self.llm_patcher = patch('geo_gpt.geocoder.get_llm')
        self.mock_llm = self.llm_patcher.start()
        self.mock_llm.return_value = MagicMock()
        
        # Create a geocoder instance for tests
        self.geocoder = GeoCoder()
        
    def tearDown(self):
        """Clean up after tests"""
        self.llm_patcher.stop()
    
    def test_initialize_geocoder(self):
        """Test that GeoCoder can be initialized without error"""
        geocoder = GeoCoder()
        self.assertIsNotNone(geocoder)
        
    @patch('pgeocode.Nominatim')
    def test_geocode_with_pgeocode(self, mock_nominatim):
        """Test geocoding with pgeocode"""
        # Create a mock result from pgeocode
        mock_result = pd.Series({
            'country_code': 'US',
            'postal_code': '90210',
            'place_name': 'Beverly Hills',
            'state_name': 'California',
            'state_code': 'CA',
            'county_name': 'Los Angeles',
            'latitude': 34.0901,
            'longitude': -118.4065,
        })
        
        # Set up the mock to return our test data
        mock_nominatim_instance = MagicMock()
        mock_nominatim_instance.query_postal_code.return_value = mock_result
        mock_nominatim.return_value = mock_nominatim_instance
        
        # Run the geocoding
        location = self.geocoder.geocode(
            zip_code="90210",
            country="US",
            use_llm=False  # Don't use LLM fallback for this test
        )
        
        # Check that the result is as expected
        self.assertIsInstance(location, GeoLocation)
        self.assertIn(location.country, ["USA", "US"])  # Depending on pycountry availability
        self.assertEqual(location.postal_code, "90210")
        self.assertEqual(location.city, "Beverly Hills")
        self.assertEqual(location.latitude, 34.0901)
        self.assertEqual(location.longitude, -118.4065)
        
    @patch('geo_gpt.geocoder.GeoCoder._geocode_with_pgeocode')
    def test_llm_fallback(self, mock_geocode_with_pgeocode):
        """Test that LLM fallback works when pgeocode fails"""
        # Make pgeocode return None to trigger fallback
        mock_geocode_with_pgeocode.return_value = None
        
        # Create a mock result from the LLM
        mock_llm_result = GeoLocation(
            country="USA",
            country_full="United States",
            postal_code="55501",
            city="Mankato",
            state_full="Minnesota",
            state_code="MN",
            latitude=44.1635,
            longitude=-93.9994,
            accuracy="high",
            formatted_address="Carlson Craft, Mankato, MN, USA"
        )
        
        # Mock the LLM geocoding to return our test data
        with patch.object(self.geocoder, '_geocode_with_llm', return_value=mock_llm_result):
            # Run the geocoding
            location = self.geocoder.geocode(
                business_name="Carlson Craft",
                state_name="Minnesota",
                country="US"
            )
            
            # Check that the result is as expected
            self.assertIsInstance(location, GeoLocation)
            self.assertEqual(location.country, "USA")
            self.assertEqual(location.city, "Mankato")
            self.assertEqual(location.state_code, "MN")
            
    def test_normalize_country_code(self):
        """Test country code normalization"""
        # Test with 2-letter code
        code = self.geocoder._normalize_country_code("us")
        self.assertEqual(code, "US")
        
        # Test with 3-letter code
        code = self.geocoder._normalize_country_code("USA")
        self.assertEqual(code, "US")
        
        # Test with full name
        code = self.geocoder._normalize_country_code("United States")
        self.assertEqual(code, "US")

    def test_haversine_distance(self):
        """Test the haversine distance calculation"""
        # New York to Los Angeles (approximately 3940 km)
        distance = self.geocoder._haversine_distance(
            40.7128, -74.0060,  # New York
            34.0522, -118.2437  # Los Angeles
        )
        self.assertGreater(distance, 3900)
        self.assertLess(distance, 4000)


if __name__ == '__main__':
    unittest.main()