"""
Metrics for evaluating causal consistency in generative models.
"""
import torch


def calculate_causal_fidelity_score(model, rule_function, num_samples=100, device="cpu"):
    """Calculate Causal Fidelity Score (CFS) to measure causal consistency.
    
    The CFS measures how often the model's generated outputs follow the specified
    causal rule when the input contains the cause.
    
    Args:
        model: Model to evaluate
        rule_function: Function that evaluates if a cause-effect relationship is maintained
            Should take (input, output) and return True if rule is followed, False otherwise
        num_samples (int): Number of samples to test
        device (str): Device to use for evaluation
            
    Returns:
        float: Causal Fidelity Score between 0 and 1
        
    Example:
        >>> def rain_wet_ground_rule(latent, image):
        ...     # Check if rain (latent[0] > 0.7) results in wet ground (dark bottom region)
        ...     rain_present = latent[0] > 0.7
        ...     ground_region = image[0, :, 48:64, :]  # Bottom region
        ...     ground_wet = ground_region.mean() < 0.4  # Dark = wet
        ...     return (not rain_present) or (rain_present and ground_wet)
        >>> 
        >>> score = calculate_causal_fidelity_score(model, rain_wet_ground_rule)
    """
    correct = 0
    model.eval()
    
    with torch.no_grad():
        for _ in range(num_samples):
            # Generate random input
            if hasattr(model, 'latent_dim'):
                latent = torch.randn(1, model.latent_dim, device=device)
                
                # Generate output
                output = model(latent)
                
                # Check if causal rule is satisfied
                if rule_function(latent, output):
                    correct += 1
    
    return correct / num_samples


def calculate_temporal_causal_fidelity(model, temporal_rule_function, num_samples=10, device="cpu"):
    """Calculate Temporal Causal Fidelity Score for video generation.
    
    This measures how often temporal causal rules are respected in generated videos.
    
    Args:
        model: Video generation model to evaluate
        temporal_rule_function: Function that evaluates if a temporal cause-effect relationship is maintained
            Should take (video_sequence) and return True if rule is followed, False otherwise
        num_samples (int): Number of video samples to test
        device (str): Device to use for evaluation
            
    Returns:
        float: Temporal Causal Fidelity Score between 0 and 1
    
    Example:
        >>> def hoof_dust_rule(video_sequence):
        ...     # Check if hoof impact at frame 5 causes dust at frame 6-8
        ...     dust_region = video_sequence[:, 6:9, 0, 48:, 10:30]  # Dust region in frames 6-8
        ...     initial_dust = video_sequence[:, 4, 0, 48:, 10:30].mean()  # Before impact
        ...     after_dust = dust_region.mean()  # After impact
        ...     return after_dust > initial_dust * 1.2  # Dust increased by 20%
        >>> 
        >>> score = calculate_temporal_causal_fidelity(video_model, hoof_dust_rule)
    """
    correct = 0
    model.eval()
    
    with torch.no_grad():
        for _ in range(num_samples):
            # Generate video
            if hasattr(model, 'generate_video'):
                video = model.generate_video(1, device=device)
            else:
                latent = torch.randn(1, model.latent_dim, device=device)
                video = model(latent)
            
            # Check if temporal causal rule is satisfied
            if temporal_rule_function(video):
                correct += 1
    
    return correct / num_samples


def calculate_rule_violation_rate(rule_engine, generated_texts, input_texts):
    """Calculate the rate of rule violations in generated text.
    
    Args:
        rule_engine (CausalRuleEngine): Rule engine to use for validation
        generated_texts (list): List of generated texts
        input_texts (list): List of corresponding input prompts
        
    Returns:
        dict: Statistics about rule violations
            {
                "violation_rate": float,  # Overall violation rate
                "violations_by_rule": dict,  # Count of violations per rule
                "total_samples": int  # Total number of samples
            }
    """
    total_samples = len(generated_texts)
    if total_samples == 0:
        return {"violation_rate": 0, "violations_by_rule": {}, "total_samples": 0}
    
    total_violations = 0
    violations_by_rule = {}
    
    for gen_text, input_text in zip(generated_texts, input_texts):
        violations = rule_engine.validate_output(gen_text, input_text)
        
        if violations:
            total_violations += 1
            
            for violation in violations:
                rule_name = violation.split(": ")[1] if ": " in violation else violation
                if rule_name in violations_by_rule:
                    violations_by_rule[rule_name] += 1
                else:
                    violations_by_rule[rule_name] = 1
    
    return {
        "violation_rate": total_violations / total_samples,
        "violations_by_rule": violations_by_rule,
        "total_samples": total_samples
    }