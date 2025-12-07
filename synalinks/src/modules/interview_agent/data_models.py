"""
Data models for Interview Agent system.

Defines entities and relations for the interview knowledge graph,
along with supporting data structures for conversation and PRD generation.
"""

from typing import List, Optional, Literal, Dict, Any
from datetime import datetime
from pydantic import Field

from synalinks.src.backend.pydantic.base import (
    Entity,
    Relation,
    DataModel,
)


# ============================================================================
# INTERVIEW ENTITIES
# ============================================================================


class User(Entity):
    """Individual being interviewed."""

    label: Literal["User"]
    name: str = Field(description="Full name of the user")
    role: str = Field(description="Job role or position")
    experience_level: str = Field(
        description="Experience level: beginner, intermediate, or advanced"
    )
    industry: Optional[str] = Field(
        default=None, description="Industry or domain the user works in"
    )
    team_size: Optional[int] = Field(
        default=None, description="Size of user's team if applicable"
    )


class Requirement(Entity):
    """Specific AI/automation requirement discovered during interview."""

    label: Literal["Requirement"]
    name: str = Field(description="Short name or title of the requirement")
    description: str = Field(description="Detailed description of the requirement")
    priority: int = Field(
        description="Priority level from 1 (low) to 5 (critical)", ge=1, le=5
    )
    category: str = Field(
        description="Category such as coding, data_analysis, automation, testing, documentation"
    )
    confidence_score: float = Field(
        description="Agent's confidence in this requirement (0-1)", ge=0.0, le=1.0
    )
    discovered_at: str = Field(description="ISO timestamp when requirement was discovered")


class UseCase(Entity):
    """Specific use case or workflow identified during interview."""

    label: Literal["UseCase"]
    name: str = Field(description="Name of the use case")
    description: str = Field(description="Detailed description of the workflow")
    frequency: str = Field(
        description="How often this occurs: daily, weekly, monthly, occasionally"
    )
    pain_points: List[str] = Field(
        default_factory=list, description="Specific pain points or challenges"
    )
    current_solution: Optional[str] = Field(
        default=None, description="How they currently handle this"
    )
    time_spent: Optional[str] = Field(
        default=None, description="Estimated time spent on this use case"
    )


class Technology(Entity):
    """Technology, tool, or programming language mentioned."""

    label: Literal["Technology"]
    name: str = Field(description="Name of the technology")
    type: str = Field(
        description="Type: language, framework, tool, platform, library"
    )
    proficiency: Optional[str] = Field(
        default=None, description="User's proficiency: beginner, intermediate, expert"
    )
    version: Optional[str] = Field(default=None, description="Version if specified")


class Goal(Entity):
    """User's goal or objective."""

    label: Literal["Goal"]
    name: str = Field(description="Name of the goal")
    description: str = Field(description="Detailed description of what they want to achieve")
    timeframe: Optional[str] = Field(
        default=None, description="Target timeframe for achieving this goal"
    )
    success_criteria: List[str] = Field(
        default_factory=list, description="How success will be measured"
    )
    importance: int = Field(
        description="Importance level from 1-5", ge=1, le=5, default=3
    )


class Plugin(Entity):
    """Claude Code plugin available in the marketplace."""

    label: Literal["Plugin"]
    name: str = Field(description="Plugin name")
    description: str = Field(description="What the plugin does")
    capabilities: List[str] = Field(
        default_factory=list, description="List of capabilities this plugin provides"
    )
    categories: List[str] = Field(
        default_factory=list, description="Categories this plugin belongs to"
    )
    install_count: int = Field(default=0, description="Number of installations")
    rating: float = Field(default=0.0, description="Average user rating (0-5)", ge=0.0, le=5.0)
    repository_url: Optional[str] = Field(
        default=None, description="GitHub or repository URL"
    )
    author: Optional[str] = Field(default=None, description="Plugin author")
    dependencies: List[str] = Field(
        default_factory=list, description="Required dependencies"
    )


class QuestionStrategy(Entity):
    """Tracks effectiveness of different questioning strategies."""

    label: Literal["QuestionStrategy"]
    question_type: str = Field(
        description="Type of question: open_ended, specific, follow_up, clarifying"
    )
    context_pattern: str = Field(
        description="Pattern describing when this strategy was used"
    )
    effectiveness_score: float = Field(
        description="How effective this strategy was (0-1)", ge=0.0, le=1.0
    )
    information_gain: float = Field(
        description="How much new information was discovered", ge=0.0, le=1.0
    )
    timestamp: str = Field(description="When this strategy was used")


# ============================================================================
# INTERVIEW RELATIONS
# ============================================================================


class HasRequirement(Relation):
    """User has a specific requirement."""

    subj: User
    label: Literal["HasRequirement"]
    obj: Requirement
    discovered_at: str = Field(description="When this requirement was discovered")
    context: str = Field(description="Conversation context where this was discovered")


class HasUseCase(Relation):
    """User has a specific use case."""

    subj: User
    label: Literal["HasUseCase"]
    obj: UseCase
    priority: int = Field(description="Priority of this use case", ge=1, le=5)


class HasGoal(Relation):
    """User has a specific goal."""

    subj: User
    label: Literal["HasGoal"]
    obj: Goal


class UsesTechnology(Relation):
    """User uses a specific technology."""

    subj: User
    label: Literal["UsesTechnology"]
    obj: Technology
    frequency: str = Field(description="How often they use it")


class RequirementNeedsTechnology(Relation):
    """Requirement depends on a specific technology."""

    subj: Requirement
    label: Literal["RequirementNeedsTechnology"]
    obj: Technology
    is_critical: bool = Field(description="Whether this technology is critical")


class UseCaseSupportsGoal(Relation):
    """Use case helps achieve a goal."""

    subj: UseCase
    label: Literal["UseCaseSupportsGoal"]
    obj: Goal
    impact_level: str = Field(description="Impact: high, medium, or low")


class UseCaseHasRequirement(Relation):
    """Use case requires a specific capability."""

    subj: UseCase
    label: Literal["UseCaseHasRequirement"]
    obj: Requirement


class PluginSatisfiesRequirement(Relation):
    """Plugin can satisfy a requirement."""

    subj: Plugin
    label: Literal["PluginSatisfiesRequirement"]
    obj: Requirement
    match_score: float = Field(
        description="How well the plugin matches (0-1)", ge=0.0, le=1.0
    )
    reasoning: str = Field(description="Explanation of why this plugin matches")


class PluginSupportsUseCase(Relation):
    """Plugin enables or improves a use case."""

    subj: Plugin
    label: Literal["PluginSupportsUseCase"]
    obj: UseCase
    improvement_description: str = Field(
        description="How the plugin improves this use case"
    )


class LeadsTo(Relation):
    """One requirement leads to or depends on another."""

    subj: Requirement
    label: Literal["LeadsTo"]
    obj: Requirement
    relationship_type: str = Field(
        description="Type: prerequisite, enhancement, alternative, related"
    )


class SimilarTo(Relation):
    """Two entities are similar (for pattern matching)."""

    subj: Entity
    label: Literal["SimilarTo"]
    obj: Entity
    similarity_score: float = Field(ge=0.0, le=1.0)
    similarity_basis: str = Field(
        description="What makes them similar"
    )


# ============================================================================
# CONVERSATION DATA MODELS
# ============================================================================


class InterviewQuestion(DataModel):
    """A question generated by the interview agent."""

    text: str = Field(description="The question text")
    question_type: str = Field(
        description="Type: open_ended, specific, follow_up, clarifying, concluding"
    )
    purpose: str = Field(description="What this question aims to discover")
    is_concluding: bool = Field(
        default=False, description="Whether this signals the end of interview"
    )
    expected_information: List[str] = Field(
        default_factory=list, description="What information we expect to extract"
    )


class AnalyzedResponse(DataModel):
    """Analysis of user's response to a question."""

    extracted_requirements: List[Requirement] = Field(
        default_factory=list, description="Requirements discovered in this response"
    )
    extracted_use_cases: List[UseCase] = Field(
        default_factory=list, description="Use cases discovered"
    )
    extracted_technologies: List[Technology] = Field(
        default_factory=list, description="Technologies mentioned"
    )
    extracted_goals: List[Goal] = Field(
        default_factory=list, description="Goals identified"
    )
    sentiment: str = Field(
        description="User's sentiment: positive, neutral, negative, mixed"
    )
    information_gain: float = Field(
        description="How much new information was gained (0-1)", ge=0.0, le=1.0
    )
    confidence: float = Field(
        description="Confidence in the analysis (0-1)", ge=0.0, le=1.0
    )
    follow_up_topics: List[str] = Field(
        default_factory=list, description="Topics worth exploring further"
    )


class InterviewSession(DataModel):
    """State of an ongoing interview session."""

    user: User
    messages: List[Any] = Field(
        default_factory=list, description="Conversation history"
    )
    requirements: List[Requirement] = Field(default_factory=list)
    use_cases: List[UseCase] = Field(default_factory=list)
    technologies: List[Technology] = Field(default_factory=list)
    goals: List[Goal] = Field(default_factory=list)
    started_at: str = Field(
        default_factory=lambda: datetime.now().isoformat()
    )
    current_phase: str = Field(
        default="introduction",
        description="Phase: introduction, exploration, deep_dive, validation, conclusion"
    )
    questions_asked: int = Field(default=0)


# ============================================================================
# PRD DATA MODELS
# ============================================================================


class OrganizedRequirements(DataModel):
    """Requirements organized by category and priority."""

    by_category: Dict[str, List[Requirement]] = Field(default_factory=dict)
    by_priority: Dict[int, List[Requirement]] = Field(default_factory=dict)
    all_requirements: List[Requirement] = Field(default_factory=list)


class PRDSection(DataModel):
    """A section of the PRD document."""

    title: str
    content: str
    subsections: List["PRDSection"] = Field(default_factory=list)


class PluginMatch(DataModel):
    """A plugin matched to a requirement."""

    requirement: Requirement
    plugin: Plugin
    score: float = Field(ge=0.0, le=1.0)
    reasoning: str


class PRD(DataModel):
    """Complete Product Requirements Document."""

    user: User
    executive_summary: str
    goals_objectives: str
    requirements: OrganizedRequirements
    use_cases: List[UseCase]
    technical_specification: str
    plugin_recommendations: List[PluginMatch]
    confidence_score: float = Field(
        description="Overall confidence in the PRD (0-1)", ge=0.0, le=1.0
    )
    generated_at: str = Field(
        default_factory=lambda: datetime.now().isoformat()
    )
    sections: List[PRDSection] = Field(
        default_factory=list, description="All PRD sections"
    )


class InterviewResult(DataModel):
    """Result of a complete interview session."""

    session: InterviewSession
    prd: PRD
    conversation: List[Any]
    confidence_score: float = Field(ge=0.0, le=1.0)
    total_duration: Optional[str] = Field(default=None)
    questions_asked: int


# ============================================================================
# PLUGIN MARKETPLACE DATA MODELS
# ============================================================================


class PluginRecommendation(DataModel):
    """A recommended plugin with context."""

    plugin: Plugin
    relevance_score: float = Field(ge=0.0, le=1.0)
    matched_requirements: List[Requirement]
    matched_use_cases: List[UseCase]
    reasoning: str
    installation_priority: int = Field(ge=1, le=10)


class PluginBundle(DataModel):
    """A curated bundle of plugins for a user."""

    plugins: List[Plugin]
    install_script: str
    configuration_guide: str
    estimated_setup_time: str
    coverage_score: float = Field(
        description="How well this bundle covers requirements (0-1)", ge=0.0, le=1.0
    )


# ============================================================================
# HELPER MODELS FOR AGENT OPERATIONS
# ============================================================================


class ConversationContext(DataModel):
    """Context for generating next question."""

    user: User
    messages: List[Any]
    current_requirements: List[Requirement]
    current_phase: str
    unanswered_areas: List[str] = Field(default_factory=list)


class QuestionGenerationInput(DataModel):
    """Input for question generation."""

    context: Any  # Retrieved context from knowledge base
    conversation: List[Any]
    user: User
    unanswered_areas: List[str]


class ResponseAnalysisInput(DataModel):
    """Input for analyzing user response."""

    question: InterviewQuestion
    response: str
    user: User
    conversation_history: List[Any] = Field(default_factory=list)


class KnowledgeUpdate(DataModel):
    """Update to apply to knowledge graph."""

    entities: List[Entity] = Field(default_factory=list)
    relations: List[Relation] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
