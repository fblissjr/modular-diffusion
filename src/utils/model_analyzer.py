# src/utils/helpers.py
import torch
import argparse
import json
from pathlib import Path
import safetensors.torch

def analyze_state_dict(state_dict_path):
    """Analyze state dict structure."""
    # Load state dict
    if str(state_dict_path).endswith('.safetensors'):
        state_dict = safetensors.torch.load_file(state_dict_path)
    else:
        state_dict = torch.load(state_dict_path, map_location='cpu')
    
    # Get top level keys
    top_level = {}
    for key in state_dict.keys():
        parts = key.split('.')
        if parts[0] not in top_level:
            top_level[parts[0]] = 0
        top_level[parts[0]] += 1
    
    # Get all prefixes
    prefixes = {}
    for key in state_dict.keys():
        parts = key.split('.')
        for i in range(1, len(parts) + 1):
            prefix = '.'.join(parts[:i])
            if prefix not in prefixes:
                prefixes[prefix] = 0
            prefixes[prefix] += 1
    
    # Get parameter shapes
    shapes = {}
    for key, value in state_dict.items():
        shapes[key] = list(value.shape)
    
    # Sample keys
    sample_keys = list(state_dict.keys())[:10]
    
    return {
        'top_level': top_level,
        'prefixes': prefixes,
        'shapes': {k: shapes[k] for k in sample_keys},
        'sample_keys': sample_keys,
        'total_keys': len(state_dict)
    }

def main():
    parser = argparse.ArgumentParser(description='Analyze model weights')
    parser.add_argument('path', type=str, help='Path to state dict')
    parser.add_argument('--output', type=str, default=None, help='Output file (JSON)')
    
    args = parser.parse_args()
    
    analysis = analyze_state_dict(args.path)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(analysis, f, indent=2)
    else:
        print(json.dumps(analysis, indent=2))

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

if __name__ == '__main__':
    main()