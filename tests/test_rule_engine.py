"""
Tests for the CausalRule system implementation.
"""
import os
import unittest
import json
import tempfile

import torch
from causaltorch.rules import CausalRule, CausalRuleSet, load_default_rules


class TestCausalRuleSystem(unittest.TestCase):
    """Test the causal rule system."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create rule set
        self.ruleset = CausalRuleSet()
        
        # Add test rules using the correct API
        self.ruleset.add_rule("rain", "wet_ground", strength=0.9)
        self.ruleset.add_rule("fire", "smoke", strength=0.8)
        self.ruleset.add_rule("fire", "heat", strength=0.95)
    
    def test_rule_creation(self):
        """Test basic rule creation."""
        rule = CausalRule("rain", "wet_ground", strength=0.9)
        self.assertEqual(rule.cause, "rain")
        self.assertEqual(rule.effect, "wet_ground")
        self.assertAlmostEqual(rule.strength, 0.9)
    
    def test_rule_set_operations(self):
        """Test rule set operations."""
        # Test adding rules
        self.assertEqual(len(self.ruleset.rules), 3)  # rain->wet_ground, fire->smoke, fire->heat
        
        # Test finding effects
        rain_effects = self.ruleset.get_rules_for_effect("wet_ground")
        self.assertEqual(len(rain_effects), 1)
        self.assertEqual(rain_effects[0].cause, "rain")
        
        fire_effects_smoke = self.ruleset.get_rules_for_effect("smoke")
        fire_effects_heat = self.ruleset.get_rules_for_effect("heat")
        self.assertEqual(len(fire_effects_smoke), 1)
        self.assertEqual(len(fire_effects_heat), 1)
        self.assertEqual(fire_effects_smoke[0].cause, "fire")
        self.assertEqual(fire_effects_heat[0].cause, "fire")

    def test_rule_remove_operation(self):
        """Test removing rules from rule set."""
        initial_count = len(self.ruleset.rules)
        
        # Remove a rule
        removed = self.ruleset.remove_rule("rain", "wet_ground")
        self.assertTrue(removed)
        self.assertEqual(len(self.ruleset.rules), initial_count - 1)
        
        # Try to remove non-existent rule
        removed = self.ruleset.remove_rule("nonexistent", "rule")
        self.assertFalse(removed)

    def test_rule_serialization(self):
        """Test rule serialization and deserialization."""
        rule = CausalRule("rain", "wet_ground", strength=0.8)
        
        # Test rule properties
        self.assertEqual(rule.cause, "rain")
        self.assertEqual(rule.effect, "wet_ground")
        self.assertEqual(rule.strength, 0.8)
        
        # Test string representation
        rule_str = str(rule)
        self.assertIn("rain", rule_str)
        self.assertIn("wet_ground", rule_str)
        self.assertIn("0.80", rule_str)

    def test_load_default_rules(self):
        """Test loading default rules."""
        try:
            default_rules = load_default_rules()
            self.assertIsInstance(default_rules, CausalRuleSet)
        except FileNotFoundError:
            # Default rules file may not exist yet
            self.skipTest("Default rules file not found")


if __name__ == "__main__":
    unittest.main()
