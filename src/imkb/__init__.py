"""
imkb - AI-powered incident knowledge base and root cause analysis SDK

A Python SDK for transforming incident/alert events into AI-inferrable context
and helping local or remote LLMs generate root cause analysis (RCA) and remediation suggestions.
"""

__version__ = "0.1.0"

# Core API exports
from .rca_pipeline import get_rca
from .action_pipeline import gen_playbook
from .config import ImkbConfig

__all__ = [
    "get_rca",
    "gen_playbook",
    "ImkbConfig",
    "__version__",
]

def main() -> None:
    print("Hello from imkb!")
