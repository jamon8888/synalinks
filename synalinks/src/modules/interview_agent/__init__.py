"""
Interview Agent Module

A self-evolving onboarding agent that conducts intelligent interviews using
Graph RAG to understand individual AI needs and generate Product Requirements
Documents (PRDs).

Main Components:
- InterviewAgent: Conducts multi-turn interviews with context awareness
- PRDGenerator: Generates comprehensive PRDs from interview data
- EvolutionTracker: Enables self-learning from past interactions
- PluginMarketplace: Recommends and bundles Claude Code plugins
- Data Models: Complete schema for interview knowledge graphs

Example:
    ```python
    import synalinks
    from synalinks.src.modules.interview_agent import (
        InterviewAgent,
        PRDGenerator,
        PluginMarketplace,
        create_interview_system
    )

    # Create complete interview system
    system = await create_interview_system(
        model="anthropic/claude-sonnet-4",
        kb_uri="neo4j://localhost:7687"
    )

    # Run interview
    session = InterviewSession(user=User(...))
    result = await system(session)

    # Get PRD and plugin recommendations
    print(result.prd.executive_summary)
    print(f"Recommended {len(result.prd.plugin_recommendations)} plugins")
    ```
"""

from .data_models import (
    # Entities
    User,
    Requirement,
    UseCase,
    Technology,
    Goal,
    Plugin,
    QuestionStrategy,
    # Relations
    HasRequirement,
    HasUseCase,
    HasGoal,
    UsesTechnology,
    RequirementNeedsTechnology,
    UseCaseSupportsGoal,
    UseCaseHasRequirement,
    PluginSatisfiesRequirement,
    PluginSupportsUseCase,
    LeadsTo,
    SimilarTo,
    # Interview Data Models
    InterviewQuestion,
    AnalyzedResponse,
    InterviewSession,
    ConversationContext,
    # PRD Data Models
    PRD,
    PRDSection,
    OrganizedRequirements,
    PluginMatch,
    InterviewResult,
    # Marketplace Data Models
    PluginRecommendation,
    PluginBundle,
)

from .interview_agent import InterviewAgent
from .prd_generator import PRDGenerator
from .evolution_tracker import EvolutionTracker, AdaptiveQuestionGenerator
from .plugin_marketplace import PluginMarketplace


__all__ = [
    # Main modules
    "InterviewAgent",
    "PRDGenerator",
    "EvolutionTracker",
    "AdaptiveQuestionGenerator",
    "PluginMarketplace",
    # Entities
    "User",
    "Requirement",
    "UseCase",
    "Technology",
    "Goal",
    "Plugin",
    "QuestionStrategy",
    # Relations
    "HasRequirement",
    "HasUseCase",
    "HasGoal",
    "UsesTechnology",
    "RequirementNeedsTechnology",
    "UseCaseSupportsGoal",
    "UseCaseHasRequirement",
    "PluginSatisfiesRequirement",
    "PluginSupportsUseCase",
    "LeadsTo",
    "SimilarTo",
    # Data models
    "InterviewQuestion",
    "AnalyzedResponse",
    "InterviewSession",
    "ConversationContext",
    "PRD",
    "PRDSection",
    "OrganizedRequirements",
    "PluginMatch",
    "InterviewResult",
    "PluginRecommendation",
    "PluginBundle",
    # Helper functions
    "create_interview_system",
]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


async def create_interview_system(
    model: str = "anthropic/claude-sonnet-4",
    kb_uri: str = "neo4j://localhost:7687",
    embedding_model: str = "openai/text-embedding-3-large",
    max_questions: int = 20,
):
    """
    Create a complete interview agent system.

    This is a convenience function that sets up all components:
    - Language model
    - Embedding model
    - Knowledge base with interview schema
    - Interview agent
    - PRD generator
    - Plugin marketplace

    Args:
        model: Language model identifier
        kb_uri: Knowledge base URI (Neo4j/MemGraph)
        embedding_model: Embedding model identifier
        max_questions: Maximum questions per interview

    Returns:
        Complete interview system (Synalinks Program)
    """
    import synalinks

    # Initialize models
    lm = synalinks.LanguageModel(model=model)
    emb = synalinks.EmbeddingModel(model=embedding_model)

    # Initialize knowledge base with interview schema
    kb = synalinks.KnowledgeBase(
        uri=kb_uri,
        entity_models=[
            User,
            Requirement,
            UseCase,
            Technology,
            Goal,
            Plugin,
            QuestionStrategy,
        ],
        relation_models=[
            HasRequirement,
            HasUseCase,
            HasGoal,
            UsesTechnology,
            RequirementNeedsTechnology,
            UseCaseSupportsGoal,
            UseCaseHasRequirement,
            PluginSatisfiesRequirement,
            PluginSupportsUseCase,
            LeadsTo,
            SimilarTo,
        ],
        embedding_model=emb,
    )

    # Build interview agent
    interview_agent = InterviewAgent(
        language_model=lm,
        knowledge_base=kb,
        max_questions=max_questions,
        name="interview_agent",
    )

    # Build PRD generator
    prd_gen = PRDGenerator(
        language_model=lm,
        knowledge_base=kb,
        name="prd_generator",
    )

    # Create functional program
    session_input = synalinks.Input(shape=(InterviewSession,))
    interview_result = interview_agent(session_input)

    # Generate PRD from interview result
    prd_output = prd_gen(interview_result.session)

    # Update interview result with PRD
    # (This would be done via a custom module in production)

    # Create program
    program = synalinks.Program(
        inputs=session_input,
        outputs=interview_result,
        name="InterviewAgentSystem",
    )

    return program
