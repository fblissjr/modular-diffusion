# src/core/config.py
from typing import Dict, Any, List, Optional, Union
import json
import yaml
from pathlib import Path
import copy

class ConfigManager:
    """
    Manager for hierarchical configuration.
    
    Handles loading, merging, and validating configurations
    across different levels, similar to how LLM frameworks
    handle hierarchical configuration.
    """
    
    @staticmethod
    def load_config(path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            path: Path to configuration file (JSON or YAML)
            
        Returns:
            Configuration dictionary
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
            
        with open(path, 'r') as f:
            if path.suffix.lower() in ['.json']:
                return json.load(f)
            elif path.suffix.lower() in ['.yaml', '.yml']:
                return yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config format: {path.suffix}")
    
    @staticmethod
    def merge_configs(*configs) -> Dict[str, Any]:
        """
        Merge multiple configurations with later ones taking precedence.
        
        Args:
            *configs: Configurations to merge
            
        Returns:
            Merged configuration
        """
        result = {}
        
        for config in configs:
            result = ConfigManager._deep_merge(result, config)
            
        return result
    
    @staticmethod
    def _deep_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries.
        
        Args:
            dict1: Base dictionary
            dict2: Dictionary to merge (takes precedence)
            
        Returns:
            Merged dictionary
        """
        result = copy.deepcopy(dict1)
        
        for key, value in dict2.items():
            # If both values are dicts, merge them
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigManager._deep_merge(result[key], value)
            else:
                # Otherwise just override
                result[key] = copy.deepcopy(value)
                
        return result
    
    @staticmethod
    def validate_config(config: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
        """
        Validate configuration against schema.
        
        Args:
            config: Configuration to validate
            schema: Schema definition
            
        Returns:
            List of validation errors (empty if valid)
        """
        # This is a simplistic validation - in a real implementation,
        # you might use jsonschema or similar
        errors = []
        
        # Check required fields
        for key, field_schema in schema.items():
            if field_schema.get("required", False) and key not in config:
                errors.append(f"Missing required field: {key}")
                
        # Type checking
        for key, value in config.items():
            if key in schema:
                field_schema = schema[key]
                expected_type = field_schema.get("type")
                
                if expected_type:
                    # Map schema types to Python types
                    type_map = {
                        "string": str,
                        "integer": int,
                        "number": (int, float),
                        "boolean": bool,
                        "array": list,
                        "object": dict
                    }
                    
                    python_type = type_map.get(expected_type)
                    
                    if python_type and not isinstance(value, python_type):
                        errors.append(f"Field {key} has wrong type: expected {expected_type}, got {type(value).__name__}")
        
        return errors