"""
Action Pipeline - Generate remediation actions from RCA results

Transforms root cause analysis into actionable remediation steps and playbooks.
"""

import json
import logging
from typing import Any, Optional

from .adapters.mem0 import Mem0Adapter
from .config import ImkbConfig, get_config
from .llm_client import LLMResponse, LLMRouter
from .models import ActionResult, KBItem, RCAResult
from .rca_pipeline import PromptManager, extract_json_from_text

logger = logging.getLogger(__name__)



class ActionPipeline:
    """
    Action Pipeline orchestrator

    Generates actionable remediation steps and playbooks from RCA results.
    """

    def __init__(self, config: Optional[ImkbConfig] = None):
        self.config = config or get_config()
        self.llm_router = LLMRouter(self.config)
        self.mem0_adapter = Mem0Adapter(self.config)
        self.prompt_manager = PromptManager()

    async def search_similar_actions(
        self, rca_result: RCAResult, limit: int = 5
    ) -> list[KBItem]:
        """Search for similar past actions and playbooks"""
        try:
            # Create search query from RCA
            search_query = f"actions playbook remediation {rca_result.root_cause[:100]}"

            # Search for similar actions in memory
            user_id = f"{self.config.namespace}_actions_{rca_result.extractor}"

            similar_actions = await self.mem0_adapter.search(
                query=search_query, user_id=user_id, limit=limit
            )

            logger.debug(f"Found {len(similar_actions)} similar action patterns")
            return similar_actions

        except Exception as e:
            logger.error(f"Failed to search similar actions: {e}")
            return []

    def _build_action_prompt_context(
        self, rca_result: RCAResult, similar_actions: list[KBItem]
    ) -> dict[str, Any]:
        """Build context for action generation prompt"""
        return {
            "rca_result": rca_result.to_dict(),
            "root_cause": rca_result.root_cause,
            "confidence": rca_result.confidence,
            "extractor": rca_result.extractor,
            "immediate_actions": rca_result.immediate_actions,
            "preventive_measures": rca_result.preventive_measures,
            "contributing_factors": rca_result.contributing_factors,
            "similar_actions": [action.to_dict() for action in similar_actions],
            "has_similar_actions": len(similar_actions) > 0,
        }

    def _render_action_prompt(self, context: dict[str, Any]) -> str:
        """Render action generation prompt using Jinja2 template"""
        try:
            template = self.prompt_manager.get_template(
                "action_generation/v1/template.jinja2"
            )
            return template.render(**context)
        except Exception as e:
            logger.error(f"Failed to render action prompt: {e}")
            # Fallback to basic template
            return f"""Generate actionable remediation plan for: {context['root_cause']}

Based on the analysis, provide a JSON response with actions, playbook,
priority, and other details."""

    def _parse_action_response(
        self, response: LLMResponse, rca_result: RCAResult
    ) -> ActionResult:
        """Parse LLM response into structured ActionResult"""
        try:
            # Use intelligent JSON extraction
            parsed_data = extract_json_from_text(response.content)

            if not parsed_data:
                raise ValueError("No JSON structure found in response")

            # Create ActionResult
            return ActionResult(
                actions=parsed_data.get("actions", ["Manual investigation required"]),
                playbook=parsed_data.get("playbook", "No detailed playbook generated"),
                priority=parsed_data.get("priority", "medium"),
                estimated_time=parsed_data.get("estimated_time"),
                risk_level=parsed_data.get("risk_level", "low"),
                prerequisites=parsed_data.get("prerequisites", []),
                validation_steps=parsed_data.get("validation_steps", []),
                rollback_plan=parsed_data.get("rollback_plan"),
                automation_potential=parsed_data.get("automation_potential", "manual"),
                confidence=0.8,  # Default confidence for parsed results
                metadata={
                    "llm_model": response.model,
                    "tokens_used": response.tokens_used,
                    "source_rca_extractor": rca_result.extractor,
                    "source_rca_confidence": rca_result.confidence,
                },
            )

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"Failed to parse action response: {e}")
            # Fallback action result
            return ActionResult(
                actions=rca_result.immediate_actions
                or ["Manual investigation required"],
                playbook=(
                    f"Automated playbook generation failed. "
                    f"Manual analysis needed for: {rca_result.root_cause}"
                ),
                priority="medium",
                risk_level="unknown",
                confidence=0.3,
                metadata={
                    "llm_model": response.model,
                    "parse_error": str(e),
                    "source_rca_extractor": rca_result.extractor,
                },
            )

    async def generate_actions(self, rca_result: RCAResult) -> ActionResult:
        """Generate actionable remediation plan from RCA result"""
        try:
            # Search for similar past actions
            similar_actions = await self.search_similar_actions(rca_result)

            # Build prompt context
            context = self._build_action_prompt_context(rca_result, similar_actions)

            # Render prompt
            prompt = self._render_action_prompt(context)

            # Generate LLM response
            llm_response = await self.llm_router.generate(
                prompt=prompt, template_type="action_generation"
            )

            # Parse response
            action_result = self._parse_action_response(llm_response, rca_result)

            # Store successful action plan for future reference
            await self._store_action_plan(rca_result, action_result)

            logger.info(
                f"Generated action plan with {len(action_result.actions)} actions"
            )
            return action_result

        except Exception as e:
            logger.error(f"Action generation failed: {e}")
            # Return fallback action result
            return ActionResult(
                actions=rca_result.immediate_actions
                or ["Manual investigation required"],
                playbook=(
                    f"Action generation error: {str(e)}. "
                    "Please investigate manually based on RCA findings."
                ),
                priority="medium",
                confidence=0.2,
                metadata={"error": str(e), "fallback": True},
            )

    async def _store_action_plan(
        self, rca_result: RCAResult, action_result: ActionResult
    ) -> None:
        """Store successful action plan for future learning"""
        try:
            user_id = f"{self.config.namespace}_actions_{rca_result.extractor}"

            # Create memory content
            content = f"""Action Plan for {rca_result.root_cause[:100]}:
Actions: {'; '.join(action_result.actions[:3])}
Priority: {action_result.priority}
Risk: {action_result.risk_level}
Success: {action_result.confidence > 0.7}"""

            await self.mem0_adapter.add_memory(
                content=content,
                user_id=user_id,
                metadata={
                    "type": "action_plan",
                    "extractor": rca_result.extractor,
                    "priority": action_result.priority,
                    "risk_level": action_result.risk_level,
                    "automation_potential": action_result.automation_potential,
                    "action_count": len(action_result.actions),
                },
            )

        except Exception as e:
            logger.warning(f"Failed to store action plan: {e}")


async def gen_playbook(
    rca_data: dict[str, Any], namespace: str = "default"
) -> dict[str, Any]:
    """
    Main entry point for action/playbook generation

    Args:
        rca_data: RCA result dictionary
        namespace: Namespace for multi-tenant isolation

    Returns:
        Action result dictionary
    """
    try:
        # Create RCAResult object
        rca_result = RCAResult.model_validate(rca_data)

        # Initialize pipeline with namespace
        config = get_config()
        config.namespace = namespace
        pipeline = ActionPipeline(config)

        # Generate actions
        action_result = await pipeline.generate_actions(rca_result)

        return action_result.to_dict()

    except Exception as e:
        logger.error(f"Action pipeline failed: {e}")
        return ActionResult(
            actions=["Manual investigation required due to pipeline error"],
            playbook=f"Action pipeline error: {str(e)}",
            priority="medium",
            confidence=0.1,
            metadata={"error": str(e)},
        ).to_dict()
