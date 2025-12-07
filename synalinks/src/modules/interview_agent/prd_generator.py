"""
PRD (Product Requirements Document) Generator

Generates comprehensive PRDs from interview knowledge graphs.
"""

from typing import List, Dict, Optional
from collections import defaultdict

from synalinks.src.modules.module import Module
from synalinks.src.modules.generators.generator import Generator
from synalinks.src.backend.pydantic.base import DataModel

from .data_models import (
    InterviewSession,
    PRD,
    PRDSection,
    OrganizedRequirements,
    PluginMatch,
    Requirement,
    UseCase,
    User,
    Plugin,
)


# PRD Section Templates
PRD_TEMPLATES = {
    "executive_summary": """Generate an executive summary for this PRD.

Include:
- Who the user is (role, experience level, industry)
- High-level overview of their AI automation needs
- Key goals they want to achieve
- Summary of main requirement categories
- Expected impact/benefits

Keep it concise (2-3 paragraphs) and business-focused.""",
    "goals_objectives": """Generate a Goals & Objectives section.

Structure:
1. Primary Goals (what they want to achieve)
2. Success Criteria (how they'll measure success)
3. Timeline/Priorities
4. Expected Outcomes

Use the user's own goals plus inferred objectives from their requirements.""",
    "requirements": """Generate a detailed Requirements section.

Organize by category and priority:

For each requirement include:
- ID/Name
- Description
- Priority (Critical/High/Medium/Low)
- Category
- Acceptance Criteria
- Dependencies (if any)

Format as clear, actionable requirements.""",
    "use_cases": """Generate a Use Cases section.

For each use case:
- Name & Description
- Current Workflow (as-is)
- Pain Points
- Proposed Workflow (to-be with AI assistance)
- Frequency
- Expected Time Savings

Focus on concrete scenarios.""",
    "technical_spec": """Generate a Technical Specification section.

Include:
- Technology Stack (languages, frameworks, tools they use)
- Integration Requirements
- Technical Constraints
- Performance Requirements
- Security/Privacy Considerations
- Plugin Dependencies

Be specific and implementation-focused.""",
}


class PRDGenerator(Module):
    """
    Generates comprehensive Product Requirements Documents from interview data.

    Takes an InterviewSession with accumulated knowledge and produces a
    structured PRD document with plugin recommendations.

    Args:
        language_model: Language model for generation
        knowledge_base: Graph database for context
    """

    def __init__(
        self,
        language_model,
        knowledge_base,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.language_model = language_model
        self.knowledge_base = knowledge_base

        # Sub-modules
        self.section_generator = None

    async def build(self, input_shape):
        """Initialize sub-modules."""
        # Generic section generator
        self.section_generator = Generator(
            language_model=self.language_model,
            output_type=str,
            name="section_generator",
        )

    async def call(self, session: InterviewSession) -> PRD:
        """
        Generate PRD from interview session.

        Args:
            session: Completed interview session

        Returns:
            Complete PRD document
        """
        # 1. Organize requirements
        organized_reqs = self._organize_requirements(session.requirements)

        # 2. Retrieve extended user context from graph
        user_graph = await self._retrieve_user_subgraph(session.user)

        # 3. Generate each section
        sections = []

        # Executive Summary
        exec_summary = await self._generate_section(
            section_name="executive_summary",
            session=session,
            organized_reqs=organized_reqs,
            user_graph=user_graph,
        )
        sections.append(PRDSection(title="Executive Summary", content=exec_summary))

        # Goals & Objectives
        goals_obj = await self._generate_section(
            section_name="goals_objectives",
            session=session,
            organized_reqs=organized_reqs,
            user_graph=user_graph,
        )
        sections.append(PRDSection(title="Goals & Objectives", content=goals_obj))

        # Requirements (detailed)
        requirements_text = await self._generate_section(
            section_name="requirements",
            session=session,
            organized_reqs=organized_reqs,
            user_graph=user_graph,
        )
        sections.append(PRDSection(title="Requirements", content=requirements_text))

        # Use Cases
        use_cases_text = await self._generate_section(
            section_name="use_cases",
            session=session,
            organized_reqs=organized_reqs,
            user_graph=user_graph,
        )
        sections.append(PRDSection(title="Use Cases", content=use_cases_text))

        # Technical Specification
        tech_spec = await self._generate_section(
            section_name="technical_spec",
            session=session,
            organized_reqs=organized_reqs,
            user_graph=user_graph,
        )
        sections.append(
            PRDSection(title="Technical Specification", content=tech_spec)
        )

        # 4. Generate plugin recommendations
        plugin_matches = await self._match_plugins(organized_reqs)

        # 5. Calculate confidence
        confidence = self._calculate_prd_confidence(session, user_graph)

        # 6. Build complete PRD
        prd = PRD(
            user=session.user,
            executive_summary=exec_summary,
            goals_objectives=goals_obj,
            requirements=organized_reqs,
            use_cases=session.use_cases,
            technical_specification=tech_spec,
            plugin_recommendations=plugin_matches,
            confidence_score=confidence,
            sections=sections,
        )

        return prd

    # ========================================================================
    # PRIVATE HELPER METHODS
    # ========================================================================

    def _organize_requirements(
        self, requirements: List[Requirement]
    ) -> OrganizedRequirements:
        """Organize requirements by category and priority."""
        by_category = defaultdict(list)
        by_priority = defaultdict(list)

        for req in requirements:
            by_category[req.category].append(req)
            by_priority[req.priority].append(req)

        return OrganizedRequirements(
            by_category=dict(by_category),
            by_priority=dict(by_priority),
            all_requirements=requirements,
        )

    async def _retrieve_user_subgraph(self, user: User) -> Dict:
        """Retrieve all user-related information from knowledge graph."""
        try:
            # Query for user's complete context
            # This would use the knowledge base to get related entities
            # For now, return placeholder
            result = {
                "user": user,
                "total_requirements": 0,
                "total_use_cases": 0,
                "technologies": [],
                "similar_users": [],
            }
            return result
        except Exception as e:
            # If knowledge base query fails, return minimal context
            return {"user": user}

    async def _generate_section(
        self,
        section_name: str,
        session: InterviewSession,
        organized_reqs: OrganizedRequirements,
        user_graph: Dict,
    ) -> str:
        """Generate a specific PRD section."""
        # Build context for generation
        context = self._build_section_context(
            section_name=section_name,
            session=session,
            organized_reqs=organized_reqs,
            user_graph=user_graph,
        )

        # Get template
        template = PRD_TEMPLATES.get(section_name, "")

        # Generate section
        prompt = f"""{template}

Context:
{context}

Generate the {section_name.replace('_', ' ').title()} section now:"""

        section_content = await self.section_generator(prompt)

        return section_content

    def _build_section_context(
        self,
        section_name: str,
        session: InterviewSession,
        organized_reqs: OrganizedRequirements,
        user_graph: Dict,
    ) -> str:
        """Build context string for section generation."""
        context_parts = []

        # User info
        context_parts.append(f"User: {session.user.name}")
        context_parts.append(f"Role: {session.user.role}")
        context_parts.append(f"Experience Level: {session.user.experience_level}")
        if session.user.industry:
            context_parts.append(f"Industry: {session.user.industry}")

        # Requirements summary
        if section_name in ["executive_summary", "requirements"]:
            context_parts.append(f"\nTotal Requirements: {len(organized_reqs.all_requirements)}")
            context_parts.append("\nRequirements by Category:")
            for category, reqs in organized_reqs.by_category.items():
                context_parts.append(f"  - {category}: {len(reqs)}")
                for req in reqs[:3]:  # Top 3 per category
                    context_parts.append(f"    * {req.name} (Priority {req.priority})")

        # Use cases
        if section_name in ["executive_summary", "use_cases"]:
            context_parts.append(f"\nUse Cases ({len(session.use_cases)}):")
            for uc in session.use_cases:
                context_parts.append(f"  - {uc.name} ({uc.frequency})")
                context_parts.append(f"    {uc.description}")
                if uc.pain_points:
                    context_parts.append(f"    Pain points: {', '.join(uc.pain_points[:2])}")

        # Goals
        if section_name in ["executive_summary", "goals_objectives"]:
            context_parts.append(f"\nGoals ({len(session.goals)}):")
            for goal in session.goals:
                context_parts.append(f"  - {goal.name} (Importance: {goal.importance}/5)")
                context_parts.append(f"    {goal.description}")
                if goal.success_criteria:
                    context_parts.append(
                        f"    Success: {'; '.join(goal.success_criteria)}"
                    )

        # Technologies
        if section_name in ["technical_spec"]:
            context_parts.append(f"\nTechnologies ({len(session.technologies)}):")
            for tech in session.technologies:
                prof = f" ({tech.proficiency})" if tech.proficiency else ""
                context_parts.append(f"  - {tech.name} ({tech.type}){prof}")

        return "\n".join(context_parts)

    async def _match_plugins(
        self, organized_reqs: OrganizedRequirements
    ) -> List[PluginMatch]:
        """
        Match requirements to available plugins.

        This is a placeholder implementation. In production, this would:
        1. Query knowledge base for Plugin entities
        2. Use semantic search to match requirement descriptions
        3. Score matches based on capability overlap
        4. Generate reasoning for each match
        """
        matches = []

        # For now, return empty list
        # In full implementation, would query knowledge base:
        # for req in organized_reqs.all_requirements:
        #     similar_plugins = await self.knowledge_base.query_similar(
        #         entity_type=Plugin,
        #         query_text=req.description,
        #         top_k=3
        #     )
        #     for plugin in similar_plugins:
        #         score = self._calculate_match_score(req, plugin)
        #         if score > 0.6:
        #             matches.append(PluginMatch(...))

        return matches

    def _calculate_match_score(
        self, requirement: Requirement, plugin: Plugin
    ) -> float:
        """Calculate how well a plugin matches a requirement."""
        # Placeholder scoring logic
        # Would use semantic similarity, capability matching, etc.
        score = 0.0

        # Category match
        if requirement.category in plugin.categories:
            score += 0.3

        # Capability overlap (simplified)
        req_words = set(requirement.description.lower().split())
        cap_words = set(" ".join(plugin.capabilities).lower().split())
        overlap = len(req_words & cap_words)
        score += min(0.4, overlap * 0.05)

        # Rating boost
        score += plugin.rating * 0.06  # max 0.3 for 5-star plugin

        return min(1.0, score)

    def _calculate_prd_confidence(
        self, session: InterviewSession, user_graph: Dict
    ) -> float:
        """Calculate overall confidence in the PRD."""
        # Factors:
        # 1. Number and quality of requirements
        req_score = min(1.0, len(session.requirements) / 10)
        if session.requirements:
            avg_req_confidence = sum(
                r.confidence_score for r in session.requirements
            ) / len(session.requirements)
        else:
            avg_req_confidence = 0.0

        # 2. Completeness of information
        completeness = 0.0
        if session.goals:
            completeness += 0.25
        if len(session.use_cases) >= 2:
            completeness += 0.25
        if len(session.technologies) >= 2:
            completeness += 0.25
        if len(session.requirements) >= 5:
            completeness += 0.25

        # 3. Interview length (did we ask enough questions?)
        interview_depth = min(1.0, session.questions_asked / 15)

        # Weighted combination
        confidence = (
            0.3 * avg_req_confidence
            + 0.3 * completeness
            + 0.2 * req_score
            + 0.2 * interview_depth
        )

        return round(confidence, 2)

    def format_prd_markdown(self, prd: PRD) -> str:
        """Format PRD as Markdown document."""
        lines = []

        # Header
        lines.append(f"# Product Requirements Document")
        lines.append(f"## AI Automation Needs - {prd.user.name}")
        lines.append("")
        lines.append(f"**Generated:** {prd.generated_at}")
        lines.append(f"**Confidence Score:** {prd.confidence_score:.0%}")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Sections
        for section in prd.sections:
            lines.append(f"## {section.title}")
            lines.append("")
            lines.append(section.content)
            lines.append("")
            lines.append("---")
            lines.append("")

        # Plugin Recommendations
        if prd.plugin_recommendations:
            lines.append("## Recommended Plugins")
            lines.append("")
            for i, match in enumerate(prd.plugin_recommendations, 1):
                lines.append(f"### {i}. {match.plugin.name}")
                lines.append(f"**Match Score:** {match.score:.0%}")
                lines.append(f"**Satisfies:** {match.requirement.name}")
                lines.append("")
                lines.append(match.reasoning)
                lines.append("")

        return "\n".join(lines)
