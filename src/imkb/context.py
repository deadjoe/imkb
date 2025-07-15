"""
Context management for multi-tenant namespace isolation

This module provides thread-safe context management for namespace isolation
using Python's contextvars, replacing the previous global state modification
approach that had concurrency safety issues.
"""

import contextvars
from typing import Optional

# Context variable for namespace isolation
_namespace_context: contextvars.ContextVar[str] = contextvars.ContextVar(
    "namespace", default="default"
)


def set_namespace(namespace: str) -> contextvars.Token[str]:
    """
    Set the current namespace in context
    
    Args:
        namespace: The namespace to set for the current context
        
    Returns:
        Token that can be used to reset the context
    """
    return _namespace_context.set(namespace)


def get_namespace() -> str:
    """
    Get the current namespace from context
    
    Returns:
        The current namespace, defaults to "default" if not set
    """
    return _namespace_context.get()


def reset_namespace(token: contextvars.Token[str]) -> None:
    """
    Reset the namespace context using a token
    
    Args:
        token: Token returned by set_namespace()
    """
    _namespace_context.reset(token)


class NamespaceContext:
    """Context manager for namespace isolation"""

    def __init__(self, namespace: str):
        self.namespace = namespace
        self.token: Optional[contextvars.Token[str]] = None

    def __enter__(self) -> str:
        self.token = set_namespace(self.namespace)
        return self.namespace

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.token is not None:
            reset_namespace(self.token)
