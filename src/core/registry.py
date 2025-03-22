# src/core/registry.py
from typing import Dict, Type, Any, Optional, TypeVar, Generic

T = TypeVar('T')

class Registry(Generic[T]):
    """
    Registry for component types.
    
    Maintains a mapping of component types to their implementations,
    similar to how LLM frameworks register tokenizers, models, etc.
    """
    
    _instances: Dict[Type, "Registry"] = {}
    
    def __init__(self, base_class: Type[T]):
        """
        Initialize registry for a base class.
        
        Args:
            base_class: Base class for registered components
        """
        self.base_class = base_class
        self.registry: Dict[str, Type[T]] = {}
    
    @classmethod
    def get(cls, base_class: Type[T]) -> "Registry[T]":
        """
        Get or create registry for base class.
        
        Args:
            base_class: Base class for registry
            
        Returns:
            Registry for base class
        """
        if base_class not in cls._instances:
            cls._instances[base_class] = cls(base_class)
        return cls._instances[base_class]
    
    def register(self, name: str, component_class: Type[T]) -> Type[T]:
        """
        Register component class.
        
        Args:
            name: Registration name
            component_class: Component class
            
        Returns:
            Component class for decorator usage
        """
        self.registry[name] = component_class
        return component_class
    
    def create(self, name: str, config: Dict[str, Any]) -> T:
        """
        Create component instance.
        
        Args:
            name: Component type name
            config: Component configuration
            
        Returns:
            Component instance
        """
        if name not in self.registry:
            raise KeyError(f"No component registered as '{name}'")
        return self.registry[name](config)
    
    def list_available(self) -> Dict[str, Type[T]]:
        """
        Get available component types.
        
        Returns:
            Dictionary mapping names to classes
        """
        return self.registry.copy()