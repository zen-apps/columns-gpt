"""
Tests for ColumnsGPT utility functions
"""

import unittest
import pandas as pd
from columns_gpt.utils import sample_dataframe, determine_type_match


class TestSampleDataframe(unittest.TestCase):
    """Tests for the sample_dataframe function."""

    def setUp(self):
        # Create a test DataFrame
        self.df = pd.DataFrame({
            "int_col": [1, 2, 3, 4, 5],
            "str_col": ["a", "b", "c", "d", "e"],
            "float_col": [1.1, 2.2, 3.3, 4.4, 5.5]
        })

    def test_sample_size(self):
        """Test that the sample size is respected."""
        samples = sample_dataframe(self.df, "int_col", sample_size=3)
        self.assertEqual(len(samples), 3)

    def test_sample_all(self):
        """Test that all values are returned when sample_size >= df length."""
        samples = sample_dataframe(self.df, "int_col", sample_size=10)
        self.assertEqual(len(samples), 5)  # The DataFrame has 5 rows

    def test_return_type(self):
        """Test that the return values are strings."""
        samples = sample_dataframe(self.df, "int_col", sample_size=2)
        for sample in samples:
            self.assertIsInstance(sample, str)


class TestDetermineTypeMatch(unittest.TestCase):
    """Tests for the determine_type_match function."""

    def test_exact_match(self):
        """Test exact matches."""
        self.assertTrue(determine_type_match("int", "int"))
        self.assertTrue(determine_type_match("float", "float"))
        self.assertTrue(determine_type_match("str", "str"))

    def test_equivalent_types(self):
        """Test equivalent types."""
        self.assertTrue(determine_type_match("int", "integer"))
        self.assertTrue(determine_type_match("int64", "int"))
        self.assertTrue(determine_type_match("str", "string"))
        self.assertTrue(determine_type_match("float", "float64"))
        self.assertTrue(determine_type_match("date", "datetime"))

    def test_case_insensitive(self):
        """Test case insensitivity."""
        self.assertTrue(determine_type_match("INT", "int"))
        self.assertTrue(determine_type_match("Float", "FLOAT"))
        self.assertTrue(determine_type_match("String", "str"))

    def test_non_matching_types(self):
        """Test non-matching types."""
        self.assertFalse(determine_type_match("int", "float"))
        self.assertFalse(determine_type_match("str", "bool"))
        self.assertFalse(determine_type_match("date", "int"))


if __name__ == "__main__":
    unittest.main()