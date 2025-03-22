# src/core/factory.py
from typing import Dict, Type, Any, Optional
from .registry import Registry
from .component import Component

class ComponentFactory:
    """
    Factory for creating pipeline components.
    
    This centralizes component creation with configuration,
    similar to how HuggingFace's AutoModel factory works.
    """
    
    @staticmethod
    def create_component(component_type: str, 
                         config: Dict[str, Any], 
                         base_class: Type = Component) -> Component:
        """
        Create component instance from configuration.
        
        Args:
            component_type: Type of component to create
            config: Component configuration
            base_class: Base class for component registry
            
        Returns:
            Component instance
        """
        registry = Registry.get(base_class)
        return registry.create(component_type, config)
    
    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> Dict[str, Component]:
        """
        Create multiple components from configuration.
        
        Args:
            config: Configuration with component sections
            
        Returns:
            Dictionary mapping component names to instances
        """
        components = {}
        
        # Create each component from its section
        for name, component_config in config.get("components", {}).items():
            component_type = component_config.get("type")
            if component_type:
                # Add name to config
                component_config["name"] = name
                components[name] = ComponentFactory.create_component(
                    component_type, component_config)
                
        return components