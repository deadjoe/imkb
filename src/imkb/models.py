"""
Core data models for imkb

Defines the central data structures like Event and KBItem using Pydantic
for robust type validation and serialization.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class KBItem(BaseModel):
    """Knowledge base item representation"""
    doc_id: str
    excerpt: str
    score: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return self.model_dump()


class Event(BaseModel):
    """Event data structure for incident/alert information"""
    id: str
    signature: str
    timestamp: str
    severity: str
    source: str
    labels: Dict[str, str]
    message: str
    raw: Dict[str, Any] = Field(default_factory=dict)
    context_hash: Optional[str] = None
    embedding_version: str = "v1.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary representation"""
        return self.model_dump()
