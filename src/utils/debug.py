# src/utils/debug.py
import logging
import torch
from typing import Dict, Any, Optional, Union
from pathlib import Path
import safetensors.torch

logger = logging.getLogger(__name__)

def extract_model_structure(model, prefix=''):
    """
    Extract model structure as a dictionary.
    
    Args:
        model: PyTorch model
        prefix: Prefix for key names
        
    Returns:
        Dictionary with model structure
    """
    structure = {}
    
    # Handle special case of ModuleDict
    if isinstance(model, nn.ModuleDict):
        for name, module in model.items():
            key = f"{prefix}.{name}" if prefix else name
            structure[key] = extract_model_structure(module, key)
    
    # Handle special case of ModuleList
    elif isinstance(model, nn.ModuleList):
        for i, module in enumerate(model):
            key = f"{prefix}.{i}" if prefix else str(i)
            structure[key] = extract_model_structure(module, key)
    
    # Handle parameters for leaf modules
    elif not list(model.children()):
        params = {}
        for name, param in model.named_parameters(recurse=False):
            params[name] = {
                'shape': list(param.shape),
                'dtype': str(param.dtype),
                'device': str(param.device)
            }
        structure['type'] = model.__class__.__name__
        structure['params'] = params
    
    # Handle composite modules
    else:
        for name, child in model.named_children():
            key = f"{prefix}.{name}" if prefix else name
            structure[key] = extract_model_structure(child, key)
    
    return structure

def compare_state_dicts(model, state_dict_path):
    """
    Compare model state dict with loaded state dict.
    
    Args:
        model: Model to compare
        state_dict_path: Path to state dict file
        
    Returns:
        Dictionary with comparison results
    """
    # Get model state dict
    model_state_dict = model.state_dict()
    model_keys = set(model_state_dict.keys())
    
    # Load state dict
    if str(state_dict_path).endswith('.safetensors'):
        try:
            state_dict = safetensors.torch.load_file(state_dict_path)
        except Exception as e:
            logger.error(f"Error loading safetensors file: {e}")
            return {"error": str(e)}
    else:
        try:
            state_dict = torch.load(state_dict_path, map_location='cpu')
        except Exception as e:
            logger.error(f"Error loading state dict: {e}")
            return {"error": str(e)}
    
    # Get state dict keys
    state_dict_keys = set(state_dict.keys())
    
    # Compare keys
    common_keys = model_keys.intersection(state_dict_keys)
    missing_keys = model_keys - state_dict_keys
    unexpected_keys = state_dict_keys - model_keys
    
    # Check for key prefixes
    key_prefixes = {}
    for key in state_dict_keys:
        parts = key.split('.')
        prefix = parts[0]
        key_prefixes[prefix] = key_prefixes.get(prefix, 0) + 1
    
    # Generate sample mappings
    sample_mappings = {}
    for ckpt_key in list(state_dict_keys)[:10]:
        # Try to map to model key
        closest_key = find_closest_key(ckpt_key, model_keys)
        sample_mappings[ckpt_key] = closest_key
    
    return {
        "common_keys_count": len(common_keys),
        "missing_keys_count": len(missing_keys),
        "unexpected_keys_count": len(unexpected_keys),
        "sample_missing_keys": list(missing_keys)[:10],
        "sample_unexpected_keys": list(unexpected_keys)[:10],
        "key_prefixes": key_prefixes,
        "sample_mappings": sample_mappings
    }

def find_closest_key(key, target_keys):
    """Find closest matching key in target_keys set."""
    # Try direct match
    if key in target_keys:
        return key
    
    # Try without model. prefix
    if key.startswith('model.'):
        key_no_prefix = key[6:]
        if key_no_prefix in target_keys:
            return key_no_prefix
    
    # Try with different prefixes
    parts = key.split('.')
    if len(parts) > 1:
        suffix = '.'.join(parts[1:])
        for target_key in target_keys:
            if target_key.endswith(suffix):
                return target_key
    
    # No match found
    return None

def print_model_info(model):
    """Print information about model structure."""
    # Print basic info
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {param_count:,}")
    
    # Print module hierarchy
    def print_modules(mod, prefix=''):
        for name, child in mod.named_children():
            logger.info(f"{prefix}{name}: {child.__class__.__name__}")
            print_modules(child, prefix + '  ')
    
    print_modules(model)