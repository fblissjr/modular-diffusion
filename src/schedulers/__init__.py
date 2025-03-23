# src/schedulers/__init__.py
from .base import Scheduler
from .flow_unipc import FlowUniPCScheduler  # explicitly meed to import this to trigger registration

__all__ = ['Scheduler', 'FlowUniPCScheduler']