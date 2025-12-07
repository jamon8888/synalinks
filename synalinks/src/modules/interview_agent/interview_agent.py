"""
Interview Agent Module

Self-evolving interview agent that conducts intelligent onboarding conversations
using Graph RAG to understand individual AI needs.
"""

from typing import Optional, List, Callable, Any
import json
from datetime import datetime

from synalinks.src.modules.module import Module
from synalinks.src.modules.generators.generator import Generator
from synalinks.src.modules.knowledge.triplet_retriever import TripletRetriever
from synalinks.src.modules.knowledge.update_knowledge import UpdateKnowledge
from synalinks.src.backend.pydantic.base import ChatMessage, ChatRole

from .data_models import (
    InterviewSession,
    InterviewQuestion,
    InterviewResult,
    AnalyzedResponse,
    ConversationContext,
    QuestionGenerationInput,
    ResponseAnalysisInput,
    KnowledgeUpdate,
    User,
    Requirement,
    UseCase,
    Technology,
    Goal,
)


class InterviewAgent(Module):
    """
    Self-evolving interview agent using Graph RAG.

    The agent conducts multi-turn conversations to discover user requirements,
    use cases, and goals. It uses graph RAG to maintain context and adapt
    questioning strategies based on what has been discovered.

    Features:
    - Dynamic question generation based on conversation context
    - Graph RAG retrieval for context-aware questioning
    - Automatic knowledge graph updates
    - Self-evolution through strategy tracking
    - Multi-phase interview flow (introduction → exploration → validation)

    Args:
        language_model: Language model for generation
        knowledge_base: Graph database for RAG
        max_questions: Maximum questions to ask (default: 20)
        adaptation_threshold: When to adapt strategy (default: 0.7)
        user_response_callback: Async function to get user responses
    """

    def __init__(
        self,
        language_model,
        knowledge_base,
        max_questions: int = 20,
        adaptation_threshold: float = 0.7,
        user_response_callback: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.language_model = language_model
        self.knowledge_base = knowledge_base
        self.max_questions = max_questions
        self.adaptation_threshold = adaptation_threshold
        self.user_response_callback = user_response_callback

        # Sub-modules (initialized in build)
        self.context_retriever = None
        self.question_generator = None
        self.response_analyzer = None
        self.knowledge_updater = None

    async def build(self, input_shape):
        """Initialize sub-modules for the interview pipeline."""

        # Context retrieval from knowledge graph
        self.context_retriever = TripletRetriever(
            knowledge_base=self.knowledge_base,
            language_model=self.language_model,
            top_k=5,
            name="context_retriever",
        )

        # Question generation
        self.question_generator = Generator(
            language_model=self.language_model,
            output_type=InterviewQuestion,
            instructions=self._get_question_instructions(),
            name="question_generator",
        )

        # Response analysis
        self.response_analyzer = Generator(
            language_model=self.language_model,
            output_type=AnalyzedResponse,
            instructions=self._get_analyzer_instructions(),
            name="response_analyzer",
        )

        # Knowledge graph updater
        self.knowledge_updater = UpdateKnowledge(
            knowledge_base=self.knowledge_base,
            name="knowledge_updater",
        )

    async def call(self, interview_session: InterviewSession) -> InterviewResult:
        """
        Conduct an interview session.

        Args:
            interview_session: Current session state with user info

        Returns:
            InterviewResult with conversation, requirements, and metadata
        """
        conversation_history = interview_session.messages.copy()
        start_time = datetime.now()

        # Introduction phase
        if not conversation_history:
            intro = self._create_introduction(interview_session.user)
            conversation_history.append(
                ChatMessage(role=ChatRole.ASSISTANT, content=intro)
            )

        # Main interview loop
        while interview_session.questions_asked < self.max_questions:
            # 1. Retrieve relevant context from knowledge graph
            context_query = self._build_context_query(
                user=interview_session.user,
                recent_messages=conversation_history[-6:],
                current_phase=interview_session.current_phase,
            )

            try:
                context = await self.context_retriever(context_query)
            except Exception as e:
                # If no context available yet, continue with empty context
                context = {"triplets": [], "message": "No prior context available"}

            # 2. Identify gaps in current knowledge
            unanswered_areas = self._identify_gaps(interview_session)

            # 3. Generate next question
            question_input = QuestionGenerationInput(
                context=context,
                conversation=conversation_history,
                user=interview_session.user,
                unanswered_areas=unanswered_areas,
            )

            question = await self.question_generator(question_input)

            # Check if interview should conclude
            if question.is_concluding or self._should_conclude(interview_session):
                closing_message = self._create_closing(interview_session)
                conversation_history.append(
                    ChatMessage(role=ChatRole.ASSISTANT, content=closing_message)
                )
                break

            # 4. Add question to conversation
            conversation_history.append(
                ChatMessage(role=ChatRole.ASSISTANT, content=question.text)
            )
            interview_session.questions_asked += 1

            # 5. Get user response
            if self.user_response_callback:
                user_response = await self.user_response_callback(question)
            else:
                # In training/testing mode, we'd have pre-defined responses
                user_response = "This is a placeholder response for testing."

            conversation_history.append(
                ChatMessage(role=ChatRole.USER, content=user_response)
            )

            # 6. Analyze response
            analysis_input = ResponseAnalysisInput(
                question=question,
                response=user_response,
                user=interview_session.user,
                conversation_history=conversation_history,
            )

            analysis = await self.response_analyzer(analysis_input)

            # 7. Update knowledge graph
            knowledge_update = self._create_knowledge_update(
                user=interview_session.user,
                analysis=analysis,
                context=f"Q: {question.text}\nA: {user_response}",
            )

            if knowledge_update.entities or knowledge_update.relations:
                await self.knowledge_updater(knowledge_update)

            # 8. Update session state
            interview_session.requirements.extend(analysis.extracted_requirements)
            interview_session.use_cases.extend(analysis.extracted_use_cases)
            interview_session.technologies.extend(analysis.extracted_technologies)
            interview_session.goals.extend(analysis.extracted_goals)

            # 9. Update interview phase
            interview_session.current_phase = self._determine_phase(
                interview_session
            )

            # 10. Check if we have enough information
            if self._has_sufficient_information(interview_session):
                break

        # Calculate total duration
        end_time = datetime.now()
        duration = str(end_time - start_time)

        # Build result
        result = InterviewResult(
            session=interview_session,
            prd=None,  # Will be generated by PRD generator
            conversation=conversation_history,
            confidence_score=self._calculate_confidence(interview_session),
            total_duration=duration,
            questions_asked=interview_session.questions_asked,
        )

        return result

    # ========================================================================
    # PRIVATE HELPER METHODS
    # ========================================================================

    def _get_question_instructions(self) -> str:
        """Instructions for question generation."""
        return """You are an expert interviewer helping understand someone's AI automation needs.

Your goal is to ask thoughtful, targeted questions that uncover:
- Specific requirements for AI assistance
- Current workflows and pain points
- Technologies and tools they use
- Goals they want to achieve
- Use cases they encounter regularly

Guidelines:
1. Ask ONE clear question at a time
2. Build on previous answers - reference what they've said
3. Start broad, then get specific based on their responses
4. Balance between open-ended exploration and specific clarification
5. Be conversational and friendly, not interrogative
6. Recognize when you have enough information

Question types to use:
- Open-ended: "Tell me about your typical workflow for..."
- Specific: "How much time do you spend on code reviews weekly?"
- Follow-up: "You mentioned X - could you elaborate on..."
- Clarifying: "When you say Y, do you mean..."
- Concluding: "Is there anything else important I should know?"

Generate a natural, contextual question that moves the interview forward."""

    def _get_analyzer_instructions(self) -> str:
        """Instructions for response analysis."""
        return """You are analyzing a user's response to extract structured information.

Your task is to carefully read their answer and extract:

1. Requirements: Specific AI/automation capabilities they need
   - Give each a clear name and detailed description
   - Assign priority (1-5) based on emphasis and pain level
   - Categorize: coding, testing, documentation, data_analysis, automation, etc.
   - Set confidence score based on how explicit they were

2. Use Cases: Specific workflows or scenarios they described
   - Name and describe the workflow
   - Note frequency (daily/weekly/monthly)
   - List pain points they mentioned
   - Record current solution if mentioned

3. Technologies: Tools, languages, frameworks they use
   - Note type (language/framework/tool/platform)
   - Record proficiency if mentioned

4. Goals: Objectives they want to achieve
   - What they want to accomplish
   - Timeframe if mentioned
   - Success criteria
   - Importance level (1-5)

5. Metadata:
   - Sentiment: positive/neutral/negative/mixed
   - Information gain: how much NEW info did we learn (0-1)
   - Confidence: how confident are you in this analysis (0-1)
   - Follow-up topics: what should we explore more

Be precise but don't over-infer. Only extract what they actually said."""

    def _build_context_query(
        self, user: User, recent_messages: List[ChatMessage], current_phase: str
    ) -> str:
        """Build a query to retrieve relevant context from knowledge graph."""
        # Convert recent conversation to text
        conversation_text = "\n".join(
            [f"{msg.role}: {msg.content}" for msg in recent_messages[-4:]]
        )

        query = f"""Find relevant information about {user.role}s in the {user.industry or 'general'} industry
who have discussed similar topics in the following conversation:

{conversation_text}

Current interview phase: {current_phase}

Look for: similar requirements, use cases, technologies, and successful interview patterns."""

        return query

    def _identify_gaps(self, session: InterviewSession) -> List[str]:
        """Identify areas that haven't been explored yet."""
        gaps = []

        # Check if we have goals
        if not session.goals:
            gaps.append("overall goals and objectives")

        # Check if we have use cases
        if len(session.use_cases) < 2:
            gaps.append("specific use cases and workflows")

        # Check if we have technical context
        if len(session.technologies) < 2:
            gaps.append("technical stack and tools currently used")

        # Check category coverage
        categories_covered = set(r.category for r in session.requirements)
        common_categories = {
            "coding",
            "testing",
            "documentation",
            "data_analysis",
            "automation",
        }
        missing_categories = common_categories - categories_covered

        if missing_categories and len(session.requirements) < 5:
            gaps.append(f"potential needs in {', '.join(list(missing_categories)[:2])}")

        # Check for pain points
        total_pain_points = sum(len(uc.pain_points) for uc in session.use_cases)
        if total_pain_points < 3:
            gaps.append("specific pain points and challenges")

        return gaps

    def _should_conclude(self, session: InterviewSession) -> bool:
        """Determine if interview should conclude."""
        # Have we asked enough questions?
        if session.questions_asked >= self.max_questions:
            return True

        # Do we have sufficient information?
        return self._has_sufficient_information(session)

    def _has_sufficient_information(self, session: InterviewSession) -> bool:
        """Check if we have enough information for a good PRD."""
        return (
            len(session.requirements) >= 5
            and len(session.use_cases) >= 2
            and len(session.goals) >= 1
            and len(session.technologies) >= 2
        )

    def _create_introduction(self, user: User) -> str:
        """Create introductory message."""
        return f"""Hi {user.name}! 👋

I'm here to help understand your AI automation needs and recommend the best Claude Code plugins for your workflow.

I'll ask you a series of questions about:
- What you're working on and your goals
- Your current workflows and pain points
- Technologies and tools you use
- Specific tasks you'd like to automate or improve

This should take about 10-15 minutes. Ready to get started?

Let's begin: What's your primary role and what kind of projects do you typically work on?"""

    def _create_closing(self, session: InterviewSession) -> str:
        """Create closing message."""
        return f"""Thank you for sharing all that information, {session.user.name}!

I've gathered {len(session.requirements)} key requirements across {len(session.use_cases)} use cases.

I'll now generate a comprehensive requirements document and recommend personalized plugins that match your needs. Give me just a moment..."""

    def _create_knowledge_update(
        self, user: User, analysis: AnalyzedResponse, context: str
    ) -> KnowledgeUpdate:
        """Create knowledge graph update from analysis."""
        from .data_models import (
            HasRequirement,
            HasUseCase,
            HasGoal,
            UsesTechnology,
            UseCaseHasRequirement,
        )

        entities = []
        relations = []

        # Add requirements
        for req in analysis.extracted_requirements:
            entities.append(req)
            relations.append(
                HasRequirement(
                    subj=user,
                    label="HasRequirement",
                    obj=req,
                    discovered_at=datetime.now().isoformat(),
                    context=context,
                )
            )

        # Add use cases
        for uc in analysis.extracted_use_cases:
            entities.append(uc)
            relations.append(
                HasUseCase(
                    subj=user,
                    label="HasUseCase",
                    obj=uc,
                    priority=3,  # Default, could be inferred
                )
            )

        # Add technologies
        for tech in analysis.extracted_technologies:
            entities.append(tech)
            relations.append(
                UsesTechnology(
                    subj=user,
                    label="UsesTechnology",
                    obj=tech,
                    frequency="unknown",  # Could be inferred from context
                )
            )

        # Add goals
        for goal in analysis.extracted_goals:
            entities.append(goal)
            relations.append(
                HasGoal(
                    subj=user,
                    label="HasGoal",
                    obj=goal,
                )
            )

        return KnowledgeUpdate(entities=entities, relations=relations)

    def _determine_phase(self, session: InterviewSession) -> str:
        """Determine current interview phase based on progress."""
        if session.questions_asked <= 2:
            return "introduction"
        elif session.questions_asked <= 8:
            return "exploration"
        elif session.questions_asked <= 15:
            return "deep_dive"
        elif session.questions_asked <= 18:
            return "validation"
        else:
            return "conclusion"

    def _calculate_confidence(self, session: InterviewSession) -> float:
        """Calculate overall confidence in collected information."""
        if not session.requirements:
            return 0.0

        # Average confidence of requirements
        req_confidence = sum(r.confidence_score for r in session.requirements) / len(
            session.requirements
        )

        # Completeness factor
        completeness = min(
            1.0,
            (
                len(session.requirements) / 5
                + len(session.use_cases) / 2
                + len(session.goals) / 1
                + len(session.technologies) / 2
            )
            / 4,
        )

        # Weighted combination
        confidence = 0.6 * req_confidence + 0.4 * completeness

        return round(confidence, 2)
