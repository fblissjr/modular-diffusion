# src/pipelines/base.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Union, Optional
import torch
from src.core.component import Component

class Pipeline(Component, ABC):
    """
    Base interface for pipelines.
    
    This provides a common interface for different pipeline implementations,
    similar to how LLM generation pipelines work.
    """
    
    @abstractmethod
    def __call__(self, **kwargs) -> Dict[str, Any]:
        """
        Run pipeline with given inputs.
        
        Args:
            **kwargs: Pipeline inputs
            
        Returns:
            Pipeline outputs
        """
        pass
    
    @abstractmethod
    def save_output(self, output: Any, path: str, **kwargs):
        """
        Save pipeline output to file.
        
        Args:
            output: Pipeline output to save
            path: Output file path
            **kwargs: Additional save parameters
        """
        pass