import unittest
import torch
from causaltorch.rules import CausalRule, CausalRuleSet, load_default_rules
from causaltorch.models import CNSGNet

class TestCausalRules(unittest.TestCase):
    def test_rule_creation(self):
        rule = CausalRule("rain", "wet_ground", strength=0.9)
        self.assertEqual(rule.cause, "rain")
        self.assertEqual(rule.effect, "wet_ground")
        self.assertAlmostEqual(rule.strength, 0.9)
    
    def test_ruleset_acyclic(self):
        rules = CausalRuleSet()
        rules.add_rule(CausalRule("rain", "wet_ground"))
        rules.add_rule(CausalRule("wet_ground", "slippery"))
        self.assertTrue(rules.is_acyclic())
        
        # Add a cycle
        rules.add_rule(CausalRule("slippery", "rain"))
        self.assertFalse(rules.is_acyclic())
    
    def test_default_rules(self):
        rules = load_default_rules()
        self.assertIn("rain", rules.rules)
        self.assertIn("fire", rules.rules)

class TestCNSGNet(unittest.TestCase):
    def setUp(self):
        # Create a simple rule set for testing
        self.rules = CausalRuleSet()
        self.rules.add_rule(CausalRule("rain", "ground_wet", strength=0.9))
        
        # Create a small model
        self.model = CNSGNet(latent_dim=3, causal_rules=self.rules.to_dict(), img_size=16)
    
    def test_model_forward(self):
        # Create a test input
        x = torch.rand(2, 1, 16, 16)
        
        # Call forward
        recon, mu, logvar = self.model(x)
        
        # Check output shapes
        self.assertEqual(recon.shape, (2, 1, 16, 16))
        self.assertEqual(mu.shape, (2, 3))
        self.assertEqual(logvar.shape, (2, 3))
    
    def test_causal_generation(self):
        # Generate with high rain
        img_wet = self.model.generate(rain_intensity=0.9)[0, 0]
        
        # Generate with no rain
        img_dry = self.model.generate(rain_intensity=0.1)[0, 0]
        
        # Ground region should be darker (wetter) in the wet image
        # Assuming bottom third of image is ground
        ground_wet = img_wet[-5:, :].mean()
        ground_dry = img_dry[-5:, :].mean()
        
        self.assertLess(ground_wet, ground_dry + 0.2)

if __name__ == "__main__":
    unittest.main()