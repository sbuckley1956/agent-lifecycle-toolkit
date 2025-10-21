from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum

from altk.core.llm import LLMClient
from altk.core.toolkit import ComponentInput, ComponentOutput
from altk.pre_tool.refraction.src.schemas.results import (
    DebuggingResult,
)
from altk.pre_tool.refraction.src.schemas.mappings import Mapping
from nestful.schemas.api import Catalog


class SPARCReflectionDecision(str, Enum):
    """Decision made by the reflection pipeline."""

    APPROVE = "approve"
    REJECT = "reject"
    ERROR = "error"


class SPARCReflectionIssueType(str, Enum):
    """Types of issues that can be identified by reflection."""

    STATIC = "static"
    SEMANTIC_GENERAL = "semantic_general"
    SEMANTIC_FUNCTION = "semantic_function"
    SEMANTIC_PARAMETER = "semantic_parameter"
    TRANSFORM = "transform"
    ERROR = "error"


class SPARCReflectionIssue(BaseModel):
    """Represents an issue identified during reflection."""

    issue_type: SPARCReflectionIssueType
    metric_name: str
    explanation: str
    correction: Optional[Dict[str, Any]] = None


class SPARCReflectionResult(BaseModel):
    """Result of reflecting on a single tool call."""

    decision: SPARCReflectionDecision
    issues: List[SPARCReflectionIssue] = Field(default_factory=list)

    @property
    def has_issues(self) -> bool:
        """Check if any issues were found."""
        return len(self.issues) > 0


class PreToolReflectionRunInput(ComponentInput):
    tool_specs: list[dict[str, Any]] = Field(
        description="List of available tool specifications"
    )
    tool_calls: list[dict[str, Any]] = Field(
        description="List of tool calls to reflect upon"
    )


class PreToolReflectionRunOutput(ComponentOutput):
    pass


class PreToolReflectionBuildInput(ComponentInput):
    pass


class PreToolReflectionBuildOutput(ComponentOutput):
    pass


class SPARCReflectionRunInput(PreToolReflectionRunInput):
    """Input for running SPARC reflection."""

    pass


class SPARCReflectionRunOutputSchema(BaseModel):
    """Output from SPARC reflection."""

    reflection_result: SPARCReflectionResult
    execution_time_ms: float
    raw_pipeline_result: Optional[Dict[str, Any]] = None

    def should_proceed_with_tool_call(self) -> bool:
        """Determine if the tool call should proceed based on reflection."""
        return self.reflection_result.decision == SPARCReflectionDecision.APPROVE


class SPARCReflectionRunOutput(PreToolReflectionRunOutput):
    """Output for running SPARC reflection."""

    output: SPARCReflectionRunOutputSchema = Field(
        default_factory=lambda: SPARCReflectionRunOutputSchema()
    )


class RefractionRunInput(PreToolReflectionRunInput):
    mappings: Optional[list[Mapping]] = None
    memory_objects: Optional[dict[str, Any]] = None
    use_given_operators_only: bool = False


class RefractionBuildInput(PreToolReflectionBuildInput):
    tool_specs: list[dict[str, Any]] | Catalog
    top_k: int = 5
    threshold: float = 0.8
    compute_maps: bool = True


class RefractionRunOutput(PreToolReflectionRunOutput):
    result: Optional[DebuggingResult] = None


class ToolGuardBuildInputMetaData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    policy_text: str = Field(description="Text of the policy document file")
    short1: bool = Field(default=True, description="Run build short or long version. ")
    validating_llm_client: LLMClient = Field(
        description="ValidatingLLMClient for build time"
    )


class ToolGuardBuildInput(ComponentInput):
    metadata: ToolGuardBuildInputMetaData = Field(
        default_factory=lambda: ToolGuardBuildInputMetaData()
    )


class ToolGuardRunInputMetaData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    tool_name: str = Field(description="Tool name")
    tool_parms: dict = Field(default={}, description="Tool parameters")
    llm_client: LLMClient = Field(description="LLMClient for build time")


class ToolGuardRunInput(ComponentInput):
    metadata: ToolGuardRunInputMetaData = Field(
        default_factory=lambda: ToolGuardRunInputMetaData()
    )


class ToolGuardRunOutputMetaData(BaseModel):
    error_message: Union[str, bool] = Field(
        description="Error string or False if no error occurred"
    )


class ToolGuardRunOutput(ComponentOutput):
    output: ToolGuardRunOutputMetaData = Field(
        default_factory=lambda: ToolGuardRunOutputMetaData()
    )
