"""
Tests for the CausalRuleEngine implementation.
"""
import os
import unittest
import json
import tempfile

import torch
from causaltorch.rules import CausalRuleEngine

class MockTokenizer:
    """Mock tokenizer for testing."""
    
    def encode(self, text, add_special_tokens=True):
        """Mock encoding that returns the ord value of each character."""
        return [ord(c) for c in text]


class TestCausalRuleEngine(unittest.TestCase):
    """Test cases for CausalRuleEngine."""
    
    def setUp(self):
        """Create a temporary rules file for testing."""
        self.temp_rules = [
            {
                "name": "test_rule",
                "pattern": "test",
                "consequences": [
                    {
                        "text": "consequence",
                        "intensity": 1.0,
                        "required": True
                    }
                ],
                "type": "test"
            },
            {
                "name": "forbidden_rule",
                "pattern": "dangerous",
                "type": "forbidden",
                "forbidden_text": "very dangerous",
                "intensity": 5.0
            }
        ]
        
        # Create temporary file
        self.temp_file = tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json")
        json.dump(self.temp_rules, self.temp_file)
        self.temp_file.close()
        
        # Create engine
        self.engine = CausalRuleEngine(self.temp_file.name)
        
    def tearDown(self):
        """Remove temporary file."""
        os.unlink(self.temp_file.name)
    
    def test_rule_loading(self):
        """Test that rules are loaded correctly."""
        self.assertEqual(len(self.engine.rules), 2)
        self.assertEqual(self.engine.rules[0]["name"], "test_rule")
        self.assertEqual(self.engine.rules[1]["type"], "forbidden")
        
    def test_apply_rules(self):
        """Test applying causal rules to attention scores."""
        # Create mock attention scores - shape: [batch, heads, seq_len, vocab_size]
        mock_attention = torch.zeros((1, 1, 1, 128))
        tokenizer = MockTokenizer()
        
        # Apply rules
        input_text = "This is a test case"  # Contains "test" -> should boost "consequence"
        modified = self.engine.apply_rules(input_text, mock_attention, tokenizer)
        
        # Check that "consequence" tokens have boosted attention
        for c in "consequence":
            self.assertGreater(modified[0, 0, 0, ord(c)], 0)
            
    def test_forbidden_rule(self):
        """Test that forbidden rules apply negative attention."""
        mock_attention = torch.zeros((1, 1, 1, 128))
        tokenizer = MockTokenizer()
        
        # Apply rules
        input_text = "This is dangerous content"  # Contains "dangerous"
        modified = self.engine.apply_rules(input_text, mock_attention, tokenizer)
        
        # Check that "very dangerous" has negative attention
        for c in "very dangerous":
            self.assertLess(modified[0, 0, 0, ord(c)], 0)
            
    def test_validate_output(self):
        """Test output validation against causal rules."""
        # Case 1: Rule violated (contains trigger but not consequence)
        input_text = "This is a test case"
        output_text = "This is a response without the required word"
        violations = self.engine.validate_output(output_text, input_text)
        self.assertEqual(len(violations), 1)
        
        # Case 2: Rule followed (contains both trigger and consequence)
        output_text = "This is a consequence of the test"
        violations = self.engine.validate_output(output_text, input_text)
        self.assertEqual(len(violations), 0)


if __name__ == "__main__":
    unittest.main()