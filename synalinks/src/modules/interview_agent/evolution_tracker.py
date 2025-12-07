"""
Evolution Tracker

Tracks the effectiveness of interview strategies and enables the agent
to self-evolve by learning from past interactions.
"""

from typing import List, Dict, Optional, Tuple
from datetime import datetime
from collections import defaultdict
import json

from .data_models import (
    InterviewQuestion,
    AnalyzedResponse,
    QuestionStrategy,
    ConversationContext,
)


class EvolutionTracker:
    """
    Tracks and learns from interview interactions to improve over time.

    The tracker:
    1. Records question effectiveness (information gain, user engagement)
    2. Identifies successful question patterns
    3. Learns optimal question sequences for different user types
    4. Adapts strategy based on user characteristics

    Args:
        knowledge_base: Graph database to store learning
    """

    def __init__(self, knowledge_base):
        self.knowledge_base = knowledge_base
        self.session_metrics = defaultdict(list)

    async def record_interaction(
        self,
        question: InterviewQuestion,
        response: str,
        analysis: AnalyzedResponse,
        effectiveness: float,
        context: Optional[Dict] = None,
    ):
        """
        Record a question-response interaction with its effectiveness.

        Args:
            question: The question that was asked
            response: User's response
            analysis: Analysis of the response
            effectiveness: Effectiveness score (0-1), typically information_gain
            context: Additional context about the interaction
        """
        # Extract pattern from question
        pattern = self._extract_pattern(question, context or {})

        # Create strategy entity
        strategy = QuestionStrategy(
            label="QuestionStrategy",
            question_type=question.question_type,
            context_pattern=pattern,
            effectiveness_score=effectiveness,
            information_gain=analysis.information_gain,
            timestamp=datetime.now().isoformat(),
        )

        # Store in knowledge base
        try:
            await self.knowledge_base.add_entities([strategy])
        except Exception as e:
            # If knowledge base not ready, store in memory
            session_id = context.get("session_id", "default") if context else "default"
            self.session_metrics[session_id].append(
                {
                    "question_type": question.question_type,
                    "pattern": pattern,
                    "effectiveness": effectiveness,
                    "info_gain": analysis.information_gain,
                    "timestamp": datetime.now().isoformat(),
                }
            )

    async def get_best_strategy(
        self, current_context: ConversationContext
    ) -> Optional[str]:
        """
        Get the most effective questioning strategy for similar contexts.

        Args:
            current_context: Current conversation context

        Returns:
            Recommended question type or None
        """
        try:
            # Query for similar contexts and their outcomes
            # This would use vector similarity on context patterns
            pattern = self._extract_context_pattern(current_context)

            # For now, use simple heuristics based on phase
            phase = current_context.current_phase

            recommendations = {
                "introduction": "open_ended",
                "exploration": "specific",
                "deep_dive": "follow_up",
                "validation": "clarifying",
                "conclusion": "concluding",
            }

            return recommendations.get(phase, "open_ended")

        except Exception:
            # Fallback to default strategy
            return "open_ended"

    async def get_successful_patterns(
        self, user_type: str, industry: Optional[str] = None, top_k: int = 5
    ) -> List[Dict]:
        """
        Get successful interview patterns for similar users.

        Args:
            user_type: User role or experience level
            industry: Industry/domain
            top_k: Number of patterns to return

        Returns:
            List of successful question patterns with metadata
        """
        try:
            # Query knowledge base for successful patterns
            # Would use Cypher query to find:
            # MATCH (u:User)-[:HasRequirement]->(r:Requirement)
            # WHERE u.role = $user_type
            # WITH r, count(*) as frequency
            # ORDER BY frequency DESC
            # LIMIT $top_k

            # Placeholder: return empty for now
            return []

        except Exception:
            return []

    async def analyze_session_performance(self, session_id: str) -> Dict:
        """
        Analyze the performance of a completed interview session.

        Returns metrics like:
        - Average information gain per question
        - Question type distribution
        - Effectiveness trend over time
        - Identified gaps in coverage
        """
        if session_id not in self.session_metrics:
            return {
                "total_interactions": 0,
                "avg_effectiveness": 0.0,
                "avg_info_gain": 0.0,
                "question_type_distribution": {},
            }

        metrics = self.session_metrics[session_id]

        # Calculate statistics
        total = len(metrics)
        avg_effectiveness = (
            sum(m["effectiveness"] for m in metrics) / total if total > 0 else 0.0
        )
        avg_info_gain = (
            sum(m["info_gain"] for m in metrics) / total if total > 0 else 0.0
        )

        # Question type distribution
        type_dist = defaultdict(int)
        for m in metrics:
            type_dist[m["question_type"]] += 1

        return {
            "total_interactions": total,
            "avg_effectiveness": round(avg_effectiveness, 3),
            "avg_info_gain": round(avg_info_gain, 3),
            "question_type_distribution": dict(type_dist),
            "effectiveness_trend": [m["effectiveness"] for m in metrics],
        }

    async def suggest_next_question_type(
        self,
        conversation_history: List,
        current_requirements: List,
        unanswered_areas: List[str],
    ) -> str:
        """
        Suggest the next question type based on conversation state.

        Uses learned patterns to recommend the most effective question type.
        """
        # Analyze conversation state
        total_questions = len([m for m in conversation_history if m.role == "assistant"])

        # Early in conversation: broad, open-ended
        if total_questions < 3:
            return "open_ended"

        # If we have gaps: specific targeting
        if unanswered_areas:
            return "specific"

        # If we have some requirements but need more detail: follow-up
        if current_requirements and total_questions < 10:
            return "follow_up"

        # Later stages: clarifying
        if total_questions >= 10:
            return "clarifying"

        return "open_ended"

    def _extract_pattern(self, question: InterviewQuestion, context: Dict) -> str:
        """Extract a pattern description from question and context."""
        pattern_parts = [
            f"type:{question.question_type}",
            f"purpose:{question.purpose[:30]}",
        ]

        if context.get("phase"):
            pattern_parts.append(f"phase:{context['phase']}")

        if context.get("requirements_count"):
            pattern_parts.append(f"reqs:{context['requirements_count']}")

        return "|".join(pattern_parts)

    def _extract_context_pattern(self, context: ConversationContext) -> str:
        """Extract pattern from conversation context."""
        pattern_parts = [
            f"phase:{context.current_phase}",
            f"reqs:{len(context.current_requirements)}",
            f"industry:{context.user.industry or 'general'}",
            f"role:{context.user.role}",
        ]

        return "|".join(pattern_parts)

    async def get_learning_insights(self) -> Dict:
        """
        Get insights about what the agent has learned.

        Returns:
            Dictionary with learning statistics and insights
        """
        total_sessions = len(self.session_metrics)
        total_interactions = sum(len(m) for m in self.session_metrics.values())

        if total_interactions == 0:
            return {
                "total_sessions": 0,
                "total_interactions": 0,
                "insights": "No learning data available yet.",
            }

        # Aggregate metrics across all sessions
        all_metrics = []
        for session_metrics in self.session_metrics.values():
            all_metrics.extend(session_metrics)

        avg_effectiveness = sum(m["effectiveness"] for m in all_metrics) / len(
            all_metrics
        )
        avg_info_gain = sum(m["info_gain"] for m in all_metrics) / len(all_metrics)

        # Most effective question type
        type_effectiveness = defaultdict(list)
        for m in all_metrics:
            type_effectiveness[m["question_type"]].append(m["effectiveness"])

        best_type = max(
            type_effectiveness.items(),
            key=lambda x: sum(x[1]) / len(x[1]),
            default=("unknown", [0]),
        )

        return {
            "total_sessions": total_sessions,
            "total_interactions": total_interactions,
            "avg_effectiveness": round(avg_effectiveness, 3),
            "avg_info_gain": round(avg_info_gain, 3),
            "most_effective_question_type": best_type[0],
            "question_type_performance": {
                qtype: round(sum(scores) / len(scores), 3)
                for qtype, scores in type_effectiveness.items()
            },
        }


class AdaptiveQuestionGenerator:
    """
    Uses graph patterns and learned strategies to generate optimal questions.

    This is an advanced component that leverages the EvolutionTracker's
    learning to adaptively generate questions.
    """

    def __init__(
        self,
        language_model,
        knowledge_base,
        evolution_tracker: EvolutionTracker,
    ):
        self.language_model = language_model
        self.knowledge_base = knowledge_base
        self.evolution_tracker = evolution_tracker

    async def generate_adaptive_question(
        self,
        session,
        conversation_history: List,
        unanswered_areas: List[str],
    ) -> InterviewQuestion:
        """
        Generate a question adapted to the current context and learned patterns.

        Args:
            session: Current interview session
            conversation_history: Full conversation history
            unanswered_areas: Areas that haven't been explored

        Returns:
            Optimized InterviewQuestion
        """
        # 1. Get recommended question type from learning
        recommended_type = await self.evolution_tracker.suggest_next_question_type(
            conversation_history=conversation_history,
            current_requirements=session.requirements,
            unanswered_areas=unanswered_areas,
        )

        # 2. Find successful question patterns for similar users
        patterns = await self.evolution_tracker.get_successful_patterns(
            user_type=session.user.role,
            industry=session.user.industry,
            top_k=3,
        )

        # 3. Generate question using these insights
        # This would typically call the language model with enriched context
        # For now, return a placeholder

        return InterviewQuestion(
            text="Based on what you've told me, what would you say is your biggest bottleneck right now?",
            question_type=recommended_type,
            purpose="Identify primary pain point using learned effective patterns",
            is_concluding=False,
            expected_information=["pain_point", "priority", "current_solution"],
        )

    async def find_optimal_question_sequence(
        self, user_profile: Dict, target_requirements: int = 5
    ) -> List[str]:
        """
        Find the optimal sequence of question types for a user profile.

        Uses graph analysis to find successful paths from similar interviews.

        Args:
            user_profile: User characteristics (role, industry, experience)
            target_requirements: Desired number of requirements to discover

        Returns:
            Suggested sequence of question types
        """
        # Query graph for successful interview paths
        # MATCH path = (u:User)-[:HasRequirement*N]->(r:Requirement)
        # WHERE u.role = $role AND u.industry = $industry
        # WITH path, count(*) as success_count
        # ORDER BY success_count DESC
        # RETURN question_sequence

        # Placeholder: return a reasonable default sequence
        return [
            "open_ended",  # Start broad
            "open_ended",  # Another broad question
            "specific",  # Get specific
            "follow_up",  # Follow up on interesting points
            "specific",  # More specifics
            "follow_up",  # Deepen understanding
            "clarifying",  # Clarify ambiguities
            "specific",  # Final specific questions
            "clarifying",  # Final clarifications
            "concluding",  # Wrap up
        ]
