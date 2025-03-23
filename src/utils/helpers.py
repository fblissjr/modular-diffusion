# src/utils/helpers.py

def analyze_key_structure(model_dict, weights_dict):
    """Analyze key structure differences between model and weights."""
    # Get top-level keys
    model_top = set(k.split('.')[0] for k in model_dict.keys())
    weights_top = set(k.split('.')[0] for k in weights_dict.keys())
    
    # Compare
    missing_top = model_top - weights_top
    extra_top = weights_top - model_top
    
    # Sample keys
    model_samples = list(model_dict.keys())[:5]
    weights_samples = list(weights_dict.keys())[:5]
    
    return {
        "model_top_keys": sorted(list(model_top)),
        "weights_top_keys": sorted(list(weights_top)),
        "missing_top_keys": sorted(list(missing_top)),
        "extra_top_keys": sorted(list(extra_top)),
        "model_key_samples": model_samples,
        "weights_key_samples": weights_samples
    }