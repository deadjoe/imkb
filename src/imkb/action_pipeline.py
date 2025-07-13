"""
Action Pipeline - Generate remediation actions from RCA results

Transforms root cause analysis into actionable remediation steps and playbooks.
"""

from typing import Any, Dict, List, Optional
import json
import logging
from pathlib import Path

from .config import ImkbConfig, get_config
from .llm_client import LLMRouter, LLMResponse
from .rca_pipeline import RCAResult
from .adapters.mem0 import Mem0Adapter, KBItem

logger = logging.getLogger(__name__)


class ActionResult:
    """Action pipeline result structure"""
    
    def __init__(
        self,
        actions: List[str],
        playbook: str,
        priority: str = "medium",
        estimated_time: Optional[str] = None,
        risk_level: str = "low",
        prerequisites: Optional[List[str]] = None,
        validation_steps: Optional[List[str]] = None,
        rollback_plan: Optional[str] = None,
        automation_potential: str = "manual",
        confidence: float = 0.8,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.actions = actions
        self.playbook = playbook
        self.priority = priority
        self.estimated_time = estimated_time
        self.risk_level = risk_level
        self.prerequisites = prerequisites or []
        self.validation_steps = validation_steps or []
        self.rollback_plan = rollback_plan
        self.automation_potential = automation_potential
        self.confidence = confidence
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "actions": self.actions,
            "playbook": self.playbook,
            "priority": self.priority,
            "estimated_time": self.estimated_time,
            "risk_level": self.risk_level,
            "prerequisites": self.prerequisites,
            "validation_steps": self.validation_steps,
            "rollback_plan": self.rollback_plan,
            "automation_potential": self.automation_potential,
            "confidence": self.confidence,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActionResult":
        """Create ActionResult from dictionary"""
        return cls(
            actions=data["actions"],
            playbook=data["playbook"],
            priority=data.get("priority", "medium"),
            estimated_time=data.get("estimated_time"),
            risk_level=data.get("risk_level", "low"),
            prerequisites=data.get("prerequisites", []),
            validation_steps=data.get("validation_steps", []),
            rollback_plan=data.get("rollback_plan"),
            automation_potential=data.get("automation_potential", "manual"),
            confidence=data.get("confidence", 0.8),
            metadata=data.get("metadata", {})
        )


class ActionPipeline:
    """
    Action Pipeline orchestrator
    
    Generates actionable remediation steps and playbooks from RCA results.
    """
    
    def __init__(self, config: Optional[ImkbConfig] = None):
        self.config = config or get_config()
        self.llm_router = LLMRouter(self.config)
        self.mem0_adapter = Mem0Adapter(self.config)
    
    async def search_similar_actions(self, rca_result: RCAResult, limit: int = 5) -> List[KBItem]:
        """Search for similar past actions and playbooks"""
        try:
            # Create search query from RCA
            search_query = f"actions playbook remediation {rca_result.root_cause[:100]}"
            
            # Search for similar actions in memory
            user_id = f"{self.config.namespace}_actions_{rca_result.extractor}"
            
            similar_actions = await self.mem0_adapter.search(
                query=search_query,
                user_id=user_id,
                limit=limit
            )
            
            logger.debug(f"Found {len(similar_actions)} similar action patterns")
            return similar_actions
            
        except Exception as e:
            logger.error(f"Failed to search similar actions: {e}")
            return []
    
    def _build_action_prompt_context(self, rca_result: RCAResult, similar_actions: List[KBItem]) -> Dict[str, Any]:
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
            "has_similar_actions": len(similar_actions) > 0
        }
    
    def _render_action_prompt(self, context: Dict[str, Any]) -> str:
        """Render action generation prompt"""
        # For now, use a simple template. In production, this would use Jinja2
        prompt = f"""You are an expert Site Reliability Engineer creating actionable remediation plans.

Based on the root cause analysis provided, generate a comprehensive action plan.

## Root Cause Analysis
**Root Cause**: {context['root_cause']}
**Confidence**: {context['confidence']}
**Extractor**: {context['extractor']}

**Immediate Actions from RCA**:
{chr(10).join(f"- {action}" for action in context['immediate_actions'])}

**Preventive Measures from RCA**:
{chr(10).join(f"- {measure}" for measure in context['preventive_measures'])}

**Contributing Factors**:
{chr(10).join(f"- {factor}" for factor in context['contributing_factors'])}

{"## Similar Past Actions" if context['has_similar_actions'] else ""}
{chr(10).join(f"- {action['excerpt'][:100]}..." for action in context['similar_actions']) if context['has_similar_actions'] else ""}

## Generate Action Plan
Provide a comprehensive action plan in JSON format:

```json
{{
  "actions": [
    "Specific actionable step 1 with clear execution instructions",
    "Specific actionable step 2 with clear execution instructions",
    "Specific actionable step 3 with clear execution instructions"
  ],
  "playbook": "Detailed step-by-step playbook with commands, expected outputs, and decision points",
  "priority": "high|medium|low",
  "estimated_time": "Time estimate for completion (e.g., '15 minutes', '2 hours')",
  "risk_level": "low|medium|high",
  "prerequisites": [
    "Prerequisites needed before starting remediation",
    "Access requirements or tools needed"
  ],
  "validation_steps": [
    "How to verify the issue is resolved",
    "Metrics or indicators to check"
  ],
  "rollback_plan": "Plan to revert changes if remediation causes issues",
  "automation_potential": "manual|semi-automated|fully-automated"
}}
```

## Guidelines
1. **Be Specific**: Provide exact commands, parameters, and expected results
2. **Risk Assessment**: Consider potential impact of each action
3. **Validation**: Include verification steps to confirm resolution
4. **Safety**: Always include rollback procedures for high-risk actions
5. **Prioritization**: Order actions by urgency and impact
6. **Context Awareness**: Consider the extractor type ({context['extractor']}) and tailor actions accordingly

Generate the action plan now:"""
        
        return prompt
    
    def _parse_action_response(self, response: LLMResponse, rca_result: RCAResult) -> ActionResult:
        """Parse LLM response into structured ActionResult"""
        try:
            # Extract JSON from response
            content = response.content.strip()
            
            # Look for JSON block
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
                start_idx = content.find("{")
                end_idx = content.rfind("}") + 1
                if start_idx != -1 and end_idx > start_idx:
                    json_content = content[start_idx:end_idx]
                else:
                    raise ValueError("No JSON structure found in response")
            
            # Parse JSON
            parsed_data = json.loads(json_content)
            
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
                    "source_rca_confidence": rca_result.confidence
                }
            )
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"Failed to parse action response: {e}")
            # Fallback action result
            return ActionResult(
                actions=rca_result.immediate_actions or ["Manual investigation required"],
                playbook=f"Automated playbook generation failed. Manual analysis needed for: {rca_result.root_cause}",
                priority="medium",
                risk_level="unknown",
                confidence=0.3,
                metadata={
                    "llm_model": response.model,
                    "parse_error": str(e),
                    "source_rca_extractor": rca_result.extractor
                }
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
                prompt=prompt,
                template_type="action_generation"
            )
            
            # Parse response
            action_result = self._parse_action_response(llm_response, rca_result)
            
            # Store successful action plan for future reference
            await self._store_action_plan(rca_result, action_result)
            
            logger.info(f"Generated action plan with {len(action_result.actions)} actions")
            return action_result
            
        except Exception as e:
            logger.error(f"Action generation failed: {e}")
            # Return fallback action result
            return ActionResult(
                actions=rca_result.immediate_actions or ["Manual investigation required"],
                playbook=f"Action generation error: {str(e)}. Please investigate manually based on RCA findings.",
                priority="medium",
                confidence=0.2,
                metadata={"error": str(e), "fallback": True}
            )
    
    async def _store_action_plan(self, rca_result: RCAResult, action_result: ActionResult) -> None:
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
                    "action_count": len(action_result.actions)
                }
            )
            
        except Exception as e:
            logger.warning(f"Failed to store action plan: {e}")


async def gen_playbook(rca_data: Dict[str, Any], namespace: str = "default") -> Dict[str, Any]:
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
        rca_result = RCAResult.from_dict(rca_data)
        
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
            metadata={"error": str(e)}
        ).to_dict()