"""
RCA Pipeline - Core logic for root cause analysis

Orchestrates the flow from event input to RCA result generation.
"""

from typing import Any, Dict, List, Optional
import json
import logging
from pathlib import Path

import jinja2
import yaml

from .config import ImkbConfig, get_config
from .extractors import registry, Event, KBItem
from .llm_client import LLMRouter, LLMResponse
from .adapters.mem0 import Mem0Adapter

# Import observability if available
try:
    from .observability.tracer import trace_async, trace_operation, add_event, set_attribute
    from .observability.metrics import get_metrics
    OBSERVABILITY_AVAILABLE = True
except ImportError:
    def trace_async(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def trace_operation(*args, **kwargs):
        from contextlib import nullcontext
        return nullcontext()
    
    def add_event(*args, **kwargs):
        pass
    
    def set_attribute(*args, **kwargs):
        pass
    
    def get_metrics():
        return None
    
    OBSERVABILITY_AVAILABLE = False

logger = logging.getLogger(__name__)


class RCAResult:
    """Root Cause Analysis result structure"""
    
    def __init__(
        self,
        root_cause: str,
        confidence: float,
        extractor: str,
        references: List[KBItem],
        status: str = "SUCCESS",
        contributing_factors: Optional[List[str]] = None,
        evidence: Optional[List[str]] = None,
        immediate_actions: Optional[List[str]] = None,
        preventive_measures: Optional[List[str]] = None,
        additional_investigation: Optional[List[str]] = None,
        confidence_reasoning: Optional[str] = None,
        knowledge_gaps: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.root_cause = root_cause
        self.confidence = confidence
        self.extractor = extractor
        self.references = references
        self.status = status
        self.contributing_factors = contributing_factors or []
        self.evidence = evidence or []
        self.immediate_actions = immediate_actions or []
        self.preventive_measures = preventive_measures or []
        self.additional_investigation = additional_investigation or []
        self.confidence_reasoning = confidence_reasoning or ""
        self.knowledge_gaps = knowledge_gaps or []
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "root_cause": self.root_cause,
            "confidence": self.confidence,
            "extractor": self.extractor,
            "references": [ref.to_dict() for ref in self.references],
            "status": self.status,
            "contributing_factors": self.contributing_factors,
            "evidence": self.evidence,
            "immediate_actions": self.immediate_actions,
            "preventive_measures": self.preventive_measures,
            "additional_investigation": self.additional_investigation,
            "confidence_reasoning": self.confidence_reasoning,
            "knowledge_gaps": self.knowledge_gaps,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RCAResult":
        """Create RCAResult from dictionary"""
        references = [KBItem(doc_id=ref.get("doc_id", "unknown"), excerpt=ref.get("excerpt", ""), score=ref.get("score", 0.0), metadata=ref.get("metadata", {})) for ref in data.get("references", [])]
        return cls(
            root_cause=data["root_cause"],
            confidence=data["confidence"],
            extractor=data["extractor"],
            references=references,
            status=data.get("status", "SUCCESS"),
            contributing_factors=data.get("contributing_factors", []),
            evidence=data.get("evidence", []),
            immediate_actions=data.get("immediate_actions", []),
            preventive_measures=data.get("preventive_measures", []),
            additional_investigation=data.get("additional_investigation", []),
            confidence_reasoning=data.get("confidence_reasoning", ""),
            knowledge_gaps=data.get("knowledge_gaps", []),
            metadata=data.get("metadata", {})
        )


class PromptManager:
    """Manages Jinja2 templates for LLM prompts"""
    
    def __init__(self, prompts_dir: str = "src/imkb/prompts"):
        self.prompts_dir = Path(prompts_dir)
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(self.prompts_dir)),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
    
    def get_template(self, template_path: str) -> jinja2.Template:
        """Get Jinja2 template by path (e.g., 'test_rca/v1/template.jinja2')"""
        try:
            return self.env.get_template(template_path)
        except jinja2.TemplateNotFound:
            logger.error(f"Template not found: {template_path}")
            raise
    
    def load_template_meta(self, template_key: str) -> Dict[str, Any]:
        """Load template metadata (e.g., 'test_rca:v1' -> meta.yaml)"""
        template_name, version = template_key.split(":")
        meta_path = self.prompts_dir / template_name / version / "meta.yaml"
        
        if not meta_path.exists():
            logger.warning(f"Template metadata not found: {meta_path}")
            return {}
        
        with open(meta_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    
    def render_template(self, template_key: str, context: Dict[str, Any]) -> str:
        """Render template with context"""
        template_name, version = template_key.split(":")
        template_path = f"{template_name}/{version}/template.jinja2"
        
        template = self.get_template(template_path)
        return template.render(**context)


class RCAPipeline:
    """
    Main RCA Pipeline orchestrator
    
    Coordinates extractors, memory recall, and LLM inference to generate
    root cause analysis results.
    """
    
    def __init__(self, config: Optional[ImkbConfig] = None):
        self.config = config or get_config()
        self.llm_router = LLMRouter(self.config)
        self.prompt_manager = PromptManager()
        self._extractors = None
    
    def get_extractors(self):
        """Get enabled extractors (lazy loading)"""
        if self._extractors is None:
            self._extractors = registry.create_enabled_extractors(self.config)
            logger.info(f"Loaded {len(self._extractors)} enabled extractors")
        return self._extractors
    
    @trace_async("rca.find_matching_extractor", record_args=True)
    async def find_matching_extractor(self, event: Event):
        """Find the best extractor that matches the event (prioritize specific over generic)"""
        extractors = self.get_extractors()
        set_attribute("event.id", event.id)
        set_attribute("available_extractors", len(extractors))
        
        # Sort extractors by specificity (non-test extractors first)
        sorted_extractors = sorted(extractors, key=lambda x: 0 if x.name != "test" else 1)
        
        for extractor in sorted_extractors:
            try:
                if await extractor.match(event):
                    logger.info(f"Event {event.id} matched extractor: {extractor.name}")
                    set_attribute("matched_extractor", extractor.name)
                    add_event("extractor_matched", {"extractor": extractor.name})
                    
                    # Record metrics
                    metrics = get_metrics()
                    if metrics:
                        metrics.record_extractor_match(extractor.name)
                    
                    return extractor
            except Exception as e:
                logger.error(f"Error in extractor {extractor.name} match: {e}")
                add_event("extractor_match_error", {
                    "extractor": extractor.name,
                    "error": str(e)
                })
                continue
        
        logger.warning(f"No extractors matched event {event.id}")
        add_event("no_extractor_matched")
        return None
    
    @trace_async("rca.recall_knowledge")
    async def recall_knowledge(self, event: Event, extractor) -> List[KBItem]:
        """Recall relevant knowledge using the extractor"""
        try:
            max_results = extractor.get_max_results()
            set_attribute("extractor.name", extractor.name)
            set_attribute("extractor.max_results", max_results)
            
            knowledge_items = await extractor.recall(event, k=max_results)
            
            set_attribute("knowledge_items.count", len(knowledge_items))
            logger.info(f"Recalled {len(knowledge_items)} knowledge items for event {event.id}")
            
            # Record metrics
            metrics = get_metrics()
            if metrics:
                metrics.record_knowledge_items(extractor.name, len(knowledge_items))
            
            return knowledge_items
            
        except Exception as e:
            logger.error(f"Knowledge recall failed: {e}")
            set_attribute("recall.error", str(e))
            return []
    
    def parse_llm_response(self, response: LLMResponse, extractor_name: str, references: List[KBItem]) -> RCAResult:
        """Parse LLM response into structured RCAResult"""
        try:
            # Try to extract JSON from the response
            content = response.content.strip()
            
            # Look for JSON block in the response
            if "```json" in content:
                start_idx = content.find("```json") + 7
                end_idx = content.find("```", start_idx)
                if end_idx != -1:
                    json_content = content[start_idx:end_idx].strip()
                else:
                    json_content = content[start_idx:].strip()
            elif content.startswith("{") and content.endswith("}"):
                json_content = content
            else:
                # Fallback: look for JSON-like structure
                start_idx = content.find("{")
                end_idx = content.rfind("}") + 1
                if start_idx != -1 and end_idx > start_idx:
                    json_content = content[start_idx:end_idx]
                else:
                    raise ValueError("No JSON structure found in response")
            
            # Parse JSON
            parsed_data = json.loads(json_content)
            
            # Create RCAResult from parsed data
            return RCAResult(
                root_cause=parsed_data.get("root_cause", "Unable to determine root cause"),
                confidence=float(parsed_data.get("confidence", 0.5)),
                extractor=extractor_name,
                references=references,
                status="SUCCESS",
                contributing_factors=parsed_data.get("contributing_factors", []),
                evidence=parsed_data.get("evidence", []),
                immediate_actions=parsed_data.get("immediate_actions", []),
                preventive_measures=parsed_data.get("preventive_measures", []),
                additional_investigation=parsed_data.get("additional_investigation", []),
                confidence_reasoning=parsed_data.get("confidence_reasoning", ""),
                knowledge_gaps=parsed_data.get("knowledge_gaps", []),
                metadata={
                    "llm_model": response.model,
                    "tokens_used": response.tokens_used,
                    "raw_response": content[:500] + "..." if len(content) > 500 else content
                }
            )
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"Failed to parse LLM response: {e}")
            # Fallback to basic result
            return RCAResult(
                root_cause=f"LLM analysis completed but response parsing failed: {str(e)}",
                confidence=0.3,
                extractor=extractor_name,
                references=references,
                status="PARSE_ERROR",
                metadata={
                    "llm_model": response.model,
                    "tokens_used": response.tokens_used,
                    "parse_error": str(e),
                    "raw_response": response.content[:500] + "..." if len(response.content) > 500 else response.content
                }
            )
    
    async def generate_rca(self, event: Event, extractor, knowledge_items: List[KBItem]) -> RCAResult:
        """Generate RCA using LLM"""
        try:
            # Build prompt context
            context = extractor.get_prompt_context(event, knowledge_items)
            
            # Render prompt template
            prompt = self.prompt_manager.render_template(extractor.prompt_template, context)
            
            # Generate LLM response
            llm_response = await self.llm_router.generate(
                prompt=prompt,
                template_type="rca"  # For mock LLM
            )
            
            # Parse response into structured result
            rca_result = self.parse_llm_response(llm_response, extractor.name, knowledge_items)
            
            logger.info(f"Generated RCA for event {event.id} with confidence {rca_result.confidence}")
            return rca_result
            
        except Exception as e:
            logger.error(f"RCA generation failed: {e}")
            return RCAResult(
                root_cause=f"RCA generation failed: {str(e)}",
                confidence=0.0,
                extractor=extractor.name if extractor else "unknown",
                references=knowledge_items,
                status="LLM_ERROR",
                metadata={"error": str(e)}
            )


@trace_async("rca.get_rca", record_args=True)
async def get_rca(event_data: Dict[str, Any], namespace: str = "default") -> Dict[str, Any]:
    """
    Main entry point for RCA generation
    
    Args:
        event_data: Event dictionary with incident information
        namespace: Namespace for multi-tenant isolation
        
    Returns:
        RCA result dictionary
    """
    metrics = get_metrics()
    
    with trace_operation("rca.pipeline") as span:
        try:
            # Create Event object
            event = Event.from_dict(event_data)
            set_attribute("event.id", event.id)
            set_attribute("event.namespace", namespace)
            
            # Initialize pipeline with namespace
            config = get_config()
            config.namespace = namespace
            pipeline = RCAPipeline(config)
            
            # Find matching extractor
            extractor = await pipeline.find_matching_extractor(event)
            if not extractor:
                result = RCAResult(
                    root_cause="No suitable knowledge extractor found for this event type",
                    confidence=0.0,
                    extractor="none",
                    references=[],
                    status="NO_EXTRACTOR"
                )
                
                # Record metrics
                if metrics:
                    metrics.record_rca_request("none", namespace, "NO_EXTRACTOR")
                
                return result.to_dict()
            
            # Record successful extractor match
            set_attribute("extractor.name", extractor.name)
            
            # Recall knowledge
            knowledge_items = await pipeline.recall_knowledge(event, extractor)
            
            # Generate RCA
            rca_result = await pipeline.generate_rca(event, extractor, knowledge_items)
            
            # Record metrics
            if metrics:
                metrics.record_rca_request(extractor.name, namespace, rca_result.status)
                if rca_result.status == "SUCCESS":
                    metrics.record_rca_success(extractor.name, namespace, rca_result.confidence)
                else:
                    metrics.record_rca_error(extractor.name, namespace, rca_result.status)
            
            set_attribute("rca.confidence", rca_result.confidence)
            set_attribute("rca.status", rca_result.status)
            
            return rca_result.to_dict()
            
        except Exception as e:
            logger.error(f"RCA pipeline failed: {e}")
            
            # Record error metrics
            if metrics:
                extractor_name = getattr(extractor, 'name', 'unknown') if 'extractor' in locals() else 'unknown'
                metrics.record_rca_error(extractor_name, namespace, "PIPELINE_ERROR")
            
            result = RCAResult(
                root_cause=f"RCA pipeline error: {str(e)}",
                confidence=0.0,
                extractor="error",
                references=[],
                status="PIPELINE_ERROR",
                metadata={"error": str(e)}
            )
            
            return result.to_dict()