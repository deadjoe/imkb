"""
Core data models for imkb

Defines the central data structures like Event and KBItem using Pydantic
for robust type validation and serialization.
"""

from typing import Any, Optional

from pydantic import BaseModel, Field


class KBItem(BaseModel):
    """Knowledge base item representation"""

    doc_id: str
    excerpt: str
    score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation"""
        return self.model_dump()


class Event(BaseModel):
    """Event data structure for incident/alert information"""

    id: str
    signature: str
    timestamp: str
    severity: str
    source: str
    labels: dict[str, str]
    message: str
    raw: dict[str, Any] = Field(default_factory=dict)
    context_hash: Optional[str] = None
    embedding_version: str = "v1.0"

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary representation"""
        return self.model_dump()


class RCAResult(BaseModel):
    """Root Cause Analysis result structure"""

    root_cause: str
    confidence: float = Field(ge=0.0, le=1.0)
    extractor: str
    references: list[KBItem] = Field(default_factory=list)
    status: str = "SUCCESS"
    contributing_factors: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    immediate_actions: list[str] = Field(default_factory=list)
    preventive_measures: list[str] = Field(default_factory=list)
    additional_investigation: list[str] = Field(default_factory=list)
    confidence_reasoning: str = ""
    knowledge_gaps: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation"""
        return self.model_dump()
