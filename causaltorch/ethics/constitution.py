"""
CausalTorch Ethics Module
========================

This module implements ethical constraints and safeguards for AI models,
ensuring that generated outputs adhere to ethical principles.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import json
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
import logging


class EthicalRule:
    """Represents a single ethical rule or constraint.
    
    Each rule defines a constraint on the model's outputs, such as
    "Do not generate content that could cause harm to humans."
    
    Args:
        name (str): Unique name for the rule
        description (str): Human-readable description of the rule
        type (str): Type of rule ('prohibit', 'require', or 'guide')
        detection_fn (Callable): Function to detect rule violations
        action (str): Action to take when violation detected
            'block': Prevent generation
            'warn': Allow but log warning
            'modify': Attempt to modify output to comply
        priority (int): Priority level (higher = more important)
    """
    def __init__(
        self,
        name: str,
        description: str,
        type: str = "prohibit",
        detection_fn: Optional[Callable] = None,
        action: str = "block",
        priority: int = 1
    ):
        self.name = name
        self.description = description
        self.type = type
        self.detection_fn = detection_fn
        self.action = action
        self.priority = priority
    
    def check(self, output: Any) -> Tuple[bool, Optional[str]]:
        """Check if output violates this rule.
        
        Args:
            output: Output to check (text, image, etc.)
            
        Returns:
            Tuple[bool, Optional[str]]: (complies, reason)
                - complies: True if output complies with rule
                - reason: Reason for violation (if any)
        """
        if self.detection_fn is None:
            # Default to compliant if no detection function
            return True, None
        
        try:
            result = self.detection_fn(output)
            if isinstance(result, tuple):
                # If detection function returns (complies, reason)
                return result
            else:
                # If detection function returns bool
                return bool(result), None
        except Exception as e:
            # If detection function fails, log and assume compliance
            logging.warning(f"Error in ethical rule check '{self.name}': {e}")
            return True, f"Error in check: {str(e)}"
    
    def to_dict(self) -> Dict:
        """Convert rule to dictionary (for serialization).
        
        Returns:
            Dict: Rule as dictionary (without detection function)
        """
        return {
            "name": self.name,
            "description": self.description,
            "type": self.type,
            "action": self.action,
            "priority": self.priority
        }
    
    @classmethod
    def from_dict(cls, data: Dict, detection_fn: Optional[Callable] = None) -> 'EthicalRule':
        """Create rule from dictionary.
        
        Args:
            data (Dict): Rule data
            detection_fn (Callable, optional): Detection function
            
        Returns:
            EthicalRule: New rule instance
        """
        return cls(
            name=data["name"],
            description=data["description"],
            type=data.get("type", "prohibit"),
            detection_fn=detection_fn,
            action=data.get("action", "block"),
            priority=data.get("priority", 1)
        )


class EthicalConstitution(nn.Module):
    """Enforces ethical rules as invariants during generation.
    
    This module acts as a guardian for model outputs, ensuring they adhere
    to predefined ethical constraints. It can block, warn about, or modify
    outputs that violate ethical rules.
    
    Args:
        rules (List[EthicalRule], optional): List of ethical rules
        log_violations (bool): Whether to log rule violations
    """
    __version__ = "2.0.1"
    
    def __init__(
        self,
        rules: Optional[List[EthicalRule]] = None,
        log_violations: bool = True
    ):
        super().__init__()
        self.rules = rules or []
        self.log_violations = log_violations
        self.logger = logging.getLogger("EthicalConstitution")
        
        # Keep a record of violations
        self.violations = []
    
    def add_rule(self, rule: EthicalRule) -> None:
        """Add a new rule to the constitution.
        
        Args:
            rule (EthicalRule): Rule to add
        """
        self.rules.append(rule)
        
        # Sort rules by priority (higher first)
        self.rules.sort(key=lambda r: r.priority, reverse=True)
    
    def forward(self, generated_output: Any) -> Tuple[Any, bool, List[Dict]]:
        """Check output against ethical rules.
        
        Args:
            generated_output: Output to check
            
        Returns:
            Tuple[Any, bool, List[Dict]]: (output, passed, violations)
                - output: Original or modified output
                - passed: Whether output passed all ethical checks
                - violations: List of rule violations
        """
        output = generated_output
        passed = True
        violations = []
        
        for rule in self.rules:
            complies, reason = rule.check(output)
            
            if not complies:
                # Record violation
                violation = {
                    "rule": rule.name,
                    "reason": reason or "No reason provided",
                    "action": rule.action
                }
                violations.append(violation)
                
                if self.log_violations:
                    self.logger.warning(
                        f"Ethical violation detected: {rule.name} - {reason or 'No reason provided'}"
                    )
                
                self.violations.append(violation)
                
                # Handle violation based on rule's action
                if rule.action == "block":
                    # Mark as failed but return original output
                    passed = False
                
                elif rule.action == "modify":
                    # Attempt to modify output (not implemented yet)
                    # In a real implementation, you would use a separate
                    # model or rule-specific function to modify the output
                    pass
        
        return output, passed, violations
    
    def save_rules(self, filepath: str) -> None:
        """Save rules to a JSON file.
        
        Args:
            filepath (str): Path to save rules
        """
        rules_data = [rule.to_dict() for rule in self.rules]
        with open(filepath, 'w') as f:
            json.dump(rules_data, f, indent=2)
    
    @classmethod
    def load_rules(cls, filepath: str, detection_fns: Dict[str, Callable] = None) -> 'EthicalConstitution':
        """Load rules from a JSON file.
        
        Args:
            filepath (str): Path to load rules from
            detection_fns (Dict[str, Callable], optional): Map of rule names to detection functions
            
        Returns:
            EthicalConstitution: New constitution with loaded rules
        """
        with open(filepath, 'r') as f:
            rules_data = json.load(f)
        
        detection_fns = detection_fns or {}
        rules = []
        
        for rule_data in rules_data:
            detection_fn = detection_fns.get(rule_data["name"])
            rule = EthicalRule.from_dict(rule_data, detection_fn)
            rules.append(rule)
        
        return cls(rules=rules)


class EthicalTextFilter:
    """Filter for detecting ethical violations in text.
    
    This class provides methods to detect common ethical issues in text,
    such as harmful content, bias, or privacy violations.
    """
    
    @staticmethod
    def check_harmful_content(text: str) -> Tuple[bool, Optional[str]]:
        """Check if text contains harmful content.
        
        Args:
            text (str): Text to check
            
        Returns:
            Tuple[bool, Optional[str]]: (complies, reason)
        """
        # Simple keyword-based check (would use more sophisticated methods in production)
        harmful_patterns = [
            r"\b(kill|murder|hurt|harm|injure)\b.*\b(human|person|people|individual)",
            r"\b(make|create|build)\b.*\b(bomb|explosive|weapon|poison)",
            r"\b(how\s+to)\b.*\b(hack|steal|rob|attack)"
        ]
        
        for pattern in harmful_patterns:
            match = re.search(pattern, text.lower())
            if match:
                return False, f"Contains potentially harmful content: '{match.group(0)}'"
        
        return True, None
    
    @staticmethod
    def check_bias(text: str) -> Tuple[bool, Optional[str]]:
        """Check if text contains biased content.
        
        Args:
            text (str): Text to check
            
        Returns:
            Tuple[bool, Optional[str]]: (complies, reason)
        """
        # Simple keyword-based check for stereotypes
        bias_patterns = [
            r"\ball\s+(men|women|people\s+from|asians|africans|europeans)\s+are\b",
            r"\b(men|women)\s+can't\b",
            r"\b(racial|gender|religious)\s+stereotypes?\b"
        ]
        
        for pattern in bias_patterns:
            match = re.search(pattern, text.lower())
            if match:
                return False, f"Contains potentially biased content: '{match.group(0)}'"
        
        return True, None
    
    @staticmethod
    def check_privacy_violation(text: str) -> Tuple[bool, Optional[str]]:
        """Check if text contains privacy violations.
        
        Args:
            text (str): Text to check
            
        Returns:
            Tuple[bool, Optional[str]]: (complies, reason)
        """
        # Check for patterns that might indicate personal data
        privacy_patterns = [
            r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",  # Phone numbers
            r"\b\d{3}[-.]?\d{2}[-.]?\d{4}\b",  # SSN
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"  # Email
        ]
        
        for pattern in privacy_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return False, f"Contains private information"
        
        return True, None
    
    @staticmethod
    def check_misinformation(text: str) -> Tuple[bool, Optional[str]]:
        # This would be more sophisticated in practice
        misinfo_patterns = [
            r"\bcure for (cancer|AIDS)\b.*\b(simple|easy|quick)\b",
            r"\bvaccines cause autism\b",
            r"\bflat earth\b.*\b(proven|evidence)\b"
        ]
        
        for pattern in misinfo_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return False, f"Potential misinformation detected"
        
        return True, None


def load_default_ethical_rules() -> List[EthicalRule]:
    """Load a set of default ethical rules.
    
    Returns:
        List[EthicalRule]: Default ethical rules
    """
    rules = [
        EthicalRule(
            name="no_harm",
            description="Do not generate content that could cause harm to humans",
            type="prohibit",
            detection_fn=EthicalTextFilter.check_harmful_content,
            action="block",
            priority=10
        ),
        EthicalRule(
            name="no_bias",
            description="Avoid generating content with harmful stereotypes or bias",
            type="prohibit",
            detection_fn=EthicalTextFilter.check_bias,
            action="warn",
            priority=8
        ),
        EthicalRule(
            name="privacy",
            description="Protect private information in generated content",
            type="prohibit",
            detection_fn=EthicalTextFilter.check_privacy_violation,
            action="modify",
            priority=9
        ),
        EthicalRule(
            name="misinformation",
            description="Avoid generating known misinformation",
            type="prohibit",
            detection_fn=EthicalTextFilter.check_misinformation,
            action="block",
            priority=8
        )
    ]
    
    return rules


class EthicalLoss(nn.Module):
    """Loss function that penalizes unethical outputs.
    
    This loss adds a penalty term for outputs that violate ethical rules,
    guiding the model toward generating more ethical content during training.
    
    Args:
        constitution (EthicalConstitution): Ethical constitution
        base_loss_fn (nn.Module): Base loss function
        ethical_weight (float): Weight for ethical penalty term
    """
    def __init__(
        self,
        constitution: EthicalConstitution,
        base_loss_fn: nn.Module,
        ethical_weight: float = 1.0
    ):
        super().__init__()
        self.constitution = constitution
        self.base_loss_fn = base_loss_fn
        self.ethical_weight = ethical_weight
    
    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate loss with ethical penalty.
        
        Args:
            outputs (torch.Tensor): Model outputs
            targets (torch.Tensor): Target values
            
        Returns:
            torch.Tensor: Combined loss
        """
        # Calculate base loss
        base_loss = self.base_loss_fn(outputs, targets)
        
        # Calculate ethical penalty
        ethical_penalty = torch.tensor(0.0, device=outputs.device)
        
        # Check each output in batch
        batch_size = outputs.size(0)
        for i in range(batch_size):
            output = outputs[i].detach()  # Detach to avoid gradient issues
            
            # Convert to appropriate format for ethical check
            # This is a placeholder - actual implementation would depend on model output format
            if hasattr(output, 'cpu'):
                output = output.cpu().numpy()
            
            # Check ethical compliance
            _, passed, violations = self.constitution(output)
            
            # Add penalty for violations
            if not passed:
                ethical_penalty = ethical_penalty + self.ethical_weight * len(violations)
        
        # Normalize by batch size
        ethical_penalty = ethical_penalty / batch_size
        
        # Combine losses
        total_loss = base_loss + ethical_penalty
        
        return total_loss 