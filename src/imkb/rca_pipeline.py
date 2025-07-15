"""
RCA Pipeline - Core logic for root cause analysis

Orchestrates the flow from event input to RCA result generation.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

import jinja2
import yaml

from .config import ImkbConfig, get_config
from .extractors import registry
from .llm_client import LLMResponse, LLMRouter
from .models import Event, KBItem, RCAResult

# Import observability if available
try:
    from .observability.metrics import get_metrics
    from .observability.tracer import (
        add_event,
        set_attribute,
        trace_async,
        trace_operation,
    )

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


def extract_json_from_text(text: str) -> Optional[dict[str, Any]]:
    """
    Intelligently extract JSON from text that may contain markdown,
    explanations, or other content around the JSON.
    """
    # Strategy 1: Look for json code blocks
    json_block_pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
    matches = re.findall(json_block_pattern, text, re.DOTALL | re.IGNORECASE)

    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue

    # Strategy 2: Look for JSON objects in the text
    # Find all potential JSON objects by looking for balanced braces
    brace_stack = []
    start_pos = None

    for i, char in enumerate(text):
        if char == "{":
            if not brace_stack:
                start_pos = i
            brace_stack.append(char)
        elif char == "}":
            if brace_stack:
                brace_stack.pop()
                if not brace_stack and start_pos is not None:
                    # Found a complete JSON object
                    json_candidate = text[start_pos : i + 1]
                    try:
                        return json.loads(json_candidate)
                    except json.JSONDecodeError:
                        # Continue looking for other JSON objects
                        start_pos = None
                        continue

    # Strategy 3: Try to parse the entire text as JSON
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Strategy 4: Look for key-value patterns and construct JSON
    # This is a last resort for badly formatted responses
    patterns = {
        "root_cause": r'"?root_cause"?\s*:\s*"([^"]+)"',
        "confidence": r'"?confidence"?\s*:\s*([0-9.]+)',
        "contributing_factors": r'"?contributing_factors"?\s*:\s*\[([^\]]+)\]',
        "evidence": r'"?evidence"?\s*:\s*\[([^\]]+)\]',
        "immediate_actions": r'"?immediate_actions"?\s*:\s*\[([^\]]+)\]',
    }

    fallback_data = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            value = match.group(1)
            if key == "confidence":
                try:
                    fallback_data[key] = float(value)
                except ValueError:
                    fallback_data[key] = 0.5
            elif key in ["contributing_factors", "evidence", "immediate_actions"]:
                # Simple list parsing
                items = [item.strip().strip('"') for item in value.split(",")]
                fallback_data[key] = items
            else:
                fallback_data[key] = value

    if fallback_data:
        return fallback_data

    return None


class PromptManager:
    """Manages Jinja2 templates for LLM prompts"""

    def __init__(self, prompts_dir: Optional[str] = None):
        if prompts_dir is None:
            # Get prompts directory from config
            config = get_config()
            prompts_dir = config.prompts.prompts_dir

        self.prompts_dir = Path(prompts_dir)
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(self.prompts_dir)),
            autoescape=jinja2.select_autoescape(["html", "xml"]),
        )

    def get_template(self, template_path: str) -> jinja2.Template:
        """Get Jinja2 template by path (e.g., 'test_rca/v1/template.jinja2')"""
        try:
            return self.env.get_template(template_path)
        except jinja2.TemplateNotFound:
            logger.error(f"Template not found: {template_path}")
            raise

    def load_template_meta(self, template_key: str) -> dict[str, Any]:
        """Load template metadata (e.g., 'test_rca:v1' -> meta.yaml)"""
        template_name, version = template_key.split(":")
        meta_path = self.prompts_dir / template_name / version / "meta.yaml"

        if not meta_path.exists():
            logger.warning(f"Template metadata not found: {meta_path}")
            return {}

        with open(meta_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def render_template(self, template_key: str, context: dict[str, Any]) -> str:
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

        # Sort extractors by configured priority order
        priority_order = self.config.extractors.priority_order

        def get_priority(extractor):
            try:
                return priority_order.index(extractor.name)
            except ValueError:
                # If not in priority list, put at the end
                return len(priority_order)

        sorted_extractors = sorted(extractors, key=get_priority)

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
                add_event(
                    "extractor_match_error",
                    {"extractor": extractor.name, "error": str(e)},
                )
                continue

        logger.warning(f"No extractors matched event {event.id}")
        add_event("no_extractor_matched")
        return None

    @trace_async("rca.recall_knowledge")
    async def recall_knowledge(self, event: Event, extractor) -> list[KBItem]:
        """Recall relevant knowledge using the extractor"""
        try:
            max_results = extractor.get_max_results()
            set_attribute("extractor.name", extractor.name)
            set_attribute("extractor.max_results", max_results)

            knowledge_items = await extractor.recall(event, k=max_results)

            set_attribute("knowledge_items.count", len(knowledge_items))
            logger.info(
                f"Recalled {len(knowledge_items)} knowledge items for event {event.id}"
            )

            # Record metrics
            metrics = get_metrics()
            if metrics:
                metrics.record_knowledge_items(extractor.name, len(knowledge_items))

            return knowledge_items

        except Exception as e:
            logger.error(f"Knowledge recall failed: {e}")
            set_attribute("recall.error", str(e))
            return []

    def parse_llm_response(
        self, response: LLMResponse, extractor_name: str, references: list[KBItem]
    ) -> RCAResult:
        """Parse LLM response into structured RCAResult"""
        try:
            # Use intelligent JSON extraction
            parsed_data = extract_json_from_text(response.content)

            if not parsed_data:
                raise ValueError("No JSON structure found in response")

            # Create RCAResult from parsed data
            return RCAResult(
                root_cause=parsed_data.get(
                    "root_cause", "Unable to determine root cause"
                ),
                confidence=float(parsed_data.get("confidence", 0.5)),
                extractor=extractor_name,
                references=references,
                status="SUCCESS",
                contributing_factors=parsed_data.get("contributing_factors", []),
                evidence=parsed_data.get("evidence", []),
                immediate_actions=parsed_data.get("immediate_actions", []),
                preventive_measures=parsed_data.get("preventive_measures", []),
                additional_investigation=parsed_data.get(
                    "additional_investigation", []
                ),
                confidence_reasoning=parsed_data.get("confidence_reasoning", ""),
                knowledge_gaps=parsed_data.get("knowledge_gaps", []),
                metadata={
                    "llm_model": response.model,
                    "tokens_used": response.tokens_used,
                    "raw_response": (
                        response.content[:500] + "..."
                        if len(response.content) > 500
                        else response.content
                    ),
                },
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
                    "raw_response": (
                        response.content[:500] + "..."
                        if len(response.content) > 500
                        else response.content
                    ),
                },
            )

    async def generate_rca(
        self, event: Event, extractor, knowledge_items: list[KBItem]
    ) -> RCAResult:
        """Generate RCA using LLM"""
        try:
            # Build prompt context
            context = extractor.get_prompt_context(event, knowledge_items)

            # Render prompt template
            prompt = self.prompt_manager.render_template(
                extractor.prompt_template, context
            )

            # Generate LLM response
            llm_response = await self.llm_router.generate(
                prompt=prompt, template_type="rca"  # For mock LLM
            )

            # Parse response into structured result
            rca_result = self.parse_llm_response(
                llm_response, extractor.name, knowledge_items
            )

            logger.info(
                f"Generated RCA for event {event.id} with confidence {rca_result.confidence}"
            )
            return rca_result

        except Exception as e:
            logger.error(f"RCA generation failed: {e}")
            return RCAResult(
                root_cause=f"RCA generation failed: {str(e)}",
                confidence=0.0,
                extractor=extractor.name if extractor else "unknown",
                references=knowledge_items,
                status="LLM_ERROR",
                metadata={"error": str(e)},
            )


@trace_async("rca.get_rca", record_args=True)
async def get_rca(
    event_data: dict[str, Any], namespace: str = "default"
) -> dict[str, Any]:
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
            event = Event.model_validate(event_data)
            set_attribute("event.id", event.id)
            set_attribute("event.namespace", namespace)

            # Initialize pipeline with namespace context
            config = get_config()

            # Use context-based namespace instead of modifying global state
            from .context import NamespaceContext
            with NamespaceContext(namespace):
                pipeline = RCAPipeline(config)

                # Find matching extractor
                extractor = await pipeline.find_matching_extractor(event)
                if not extractor:
                    result = RCAResult(
                        root_cause="No suitable knowledge extractor found for this event type",
                        confidence=0.0,
                        extractor="none",
                        references=[],
                        status="NO_EXTRACTOR",
                    )

                    # Record metrics
                    if metrics:
                        metrics.record_rca_request("none", namespace, "NO_EXTRACTOR")

                    return result.model_dump()

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
                        metrics.record_rca_success(
                            extractor.name, namespace, rca_result.confidence
                        )
                    else:
                        metrics.record_rca_error(
                            extractor.name, namespace, rca_result.status
                        )

                set_attribute("rca.confidence", rca_result.confidence)
                set_attribute("rca.status", rca_result.status)

                return rca_result.model_dump()

        except Exception as e:
            logger.error(f"RCA pipeline failed: {e}")

            # Record error metrics
            if metrics:
                extractor_name = (
                    getattr(extractor, "name", "unknown")
                    if "extractor" in locals()
                    else "unknown"
                )
                metrics.record_rca_error(extractor_name, namespace, "PIPELINE_ERROR")

            result = RCAResult(
                root_cause=f"RCA pipeline error: {str(e)}",
                confidence=0.0,
                extractor="error",
                references=[],
                status="PIPELINE_ERROR",
                metadata={"error": str(e)},
            )

            return result.model_dump()
