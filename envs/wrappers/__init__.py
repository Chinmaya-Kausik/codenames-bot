"""
Environment wrappers for multi-agent Codenames.

This module provides wrappers that modify or simplify the multi-agent
environment interface for different use cases.
"""

from envs.wrappers.single_agent_wrapper import SingleAgentWrapper

__all__ = [
    "SingleAgentWrapper",
]
