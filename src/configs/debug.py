"""Debug configuration for development."""

# Debug settings that can be imported as needed
DEBUG_CONFIG = {
    # Print state dict key information during loading
    "print_state_dict_info": True,
    
    # Maximum number of keys to print
    "max_keys_to_print": 10,
    
    # Create dummy components if loading fails
    "use_dummy_on_failure": True,
    
    # Trace model forward pass
    "trace_forward": False
}

def set_debug_mode(enabled=True):
    """Enable or disable debug mode globally."""
    DEBUG_CONFIG["print_state_dict_info"] = enabled
    DEBUG_CONFIG["trace_forward"] = enabled