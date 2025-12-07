# Interview Agent Architecture - Self-Evolving Onboarding System

## Overview

An autonomous interview agent that conducts intelligent onboarding conversations using Graph RAG to understand individual AI needs and generate comprehensive Product Requirements Documents (PRDs). The system self-evolves based on user interactions and builds a knowledge graph of requirements that can be matched to Claude Code plugins.

## System Architecture

### 1. Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Interview Agent System                    │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
   ┌────▼────┐          ┌─────▼─────┐       ┌──────▼──────┐
   │Interview│          │ Graph RAG │       │PRD Generator│
   │ Agent   │◄────────►│ Knowledge │◄─────►│   Module    │
   └────┬────┘          │   Base    │       └──────┬──────┘
        │               └─────┬─────┘              │
        │                     │                    │
   ┌────▼────┐          ┌─────▼─────┐       ┌──────▼──────┐
   │Question │          │Evolution  │       │  Plugin     │
   │Strategy │          │ Tracker   │       │  Matcher    │
   └─────────┘          └───────────┘       └─────────────┘
```

### 2. Data Models (Synalinks Graph Schema)

```python
# Interview Knowledge Graph Entities

class User(synalinks.Entity):
    """Individual being interviewed"""
    label: Literal["User"]
    name: str
    role: str
    experience_level: str  # beginner, intermediate, advanced
    industry: Optional[str]

class Requirement(synalinks.Entity):
    """Specific AI/automation requirement"""
    label: Literal["Requirement"]
    name: str
    description: str
    priority: int  # 1-5
    category: str  # coding, data_analysis, automation, etc.
    confidence_score: float  # how confident the agent is about this

class UseCase(synalinks.Entity):
    """Specific use case or workflow"""
    label: Literal["UseCase"]
    name: str
    description: str
    frequency: str  # daily, weekly, monthly
    pain_points: List[str]

class Technology(synalinks.Entity):
    """Technology or tool mentioned"""
    label: Literal["Technology"]
    name: str
    type: str  # language, framework, tool, platform
    proficiency: Optional[str]

class Goal(synalinks.Entity):
    """User's goal or objective"""
    label: Literal["Goal"]
    name: str
    description: str
    timeframe: Optional[str]
    success_criteria: List[str]

class Plugin(synalinks.Entity):
    """Claude Code plugin"""
    label: Literal["Plugin"]
    name: str
    description: str
    capabilities: List[str]
    categories: List[str]
    install_count: int
    rating: float

# Relations

class HasRequirement(synalinks.Relation):
    subj: User
    label: Literal["HasRequirement"]
    obj: Requirement
    discovered_at: str  # timestamp
    context: str  # how this was discovered

class HasUseCase(synalinks.Relation):
    subj: User
    label: Literal["HasUseCase"]
    obj: UseCase
    priority: int

class RequirementNeedsTechnology(synalinks.Relation):
    subj: Requirement
    label: Literal["RequirementNeedsTechnology"]
    obj: Technology
    is_critical: bool

class UseCaseSupportsGoal(synalinks.Relation):
    subj: UseCase
    label: Literal["UseCaseSupportsGoal"]
    obj: Goal
    impact_level: str  # high, medium, low

class PluginSatisfiesRequirement(synalinks.Relation):
    subj: Plugin
    label: Literal["PluginSatisfiesRequirement"]
    obj: Requirement
    match_score: float  # 0-1
    reasoning: str

class LeadsTo(synalinks.Relation):
    """Links related requirements (discovered through interview)"""
    subj: Requirement
    label: Literal["LeadsTo"]
    obj: Requirement
    relationship_type: str  # prerequisite, enhancement, alternative
```

### 3. Interview Agent Module

```python
class InterviewAgent(synalinks.Module):
    """
    Self-evolving interview agent that conducts intelligent onboarding
    conversations using Graph RAG.

    Features:
    - Dynamic question generation based on conversation history
    - Graph RAG retrieval for context-aware questioning
    - Self-evolution through OMEGA optimization
    - Multi-turn conversation management
    - PRD generation from accumulated knowledge
    """

    def __init__(
        self,
        language_model: synalinks.LanguageModel,
        knowledge_base: synalinks.KnowledgeBase,
        max_questions: int = 20,
        adaptation_threshold: float = 0.7,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.language_model = language_model
        self.knowledge_base = knowledge_base
        self.max_questions = max_questions
        self.adaptation_threshold = adaptation_threshold

        # Sub-modules (built in build())
        self.context_retriever = None
        self.question_generator = None
        self.response_analyzer = None
        self.knowledge_updater = None
        self.evolution_tracker = None

    async def build(self, input_shape):
        # Build interview pipeline
        self.context_retriever = synalinks.modules.TripletRetriever(
            knowledge_base=self.knowledge_base,
            language_model=self.language_model,
            top_k=5
        )

        self.question_generator = synalinks.modules.Generator(
            language_model=self.language_model,
            output_type=InterviewQuestion,
            instructions=self._get_question_instructions()
        )

        self.response_analyzer = synalinks.modules.Generator(
            language_model=self.language_model,
            output_type=AnalyzedResponse,
            instructions=self._get_analyzer_instructions()
        )

        self.knowledge_updater = synalinks.modules.UpdateKnowledge(
            knowledge_base=self.knowledge_base
        )

        self.evolution_tracker = EvolutionTracker(
            knowledge_base=self.knowledge_base
        )

    async def call(self, interview_session: InterviewSession) -> InterviewResult:
        """
        Conduct an interview session.

        Args:
            interview_session: Current session state

        Returns:
            InterviewResult with conversation, extracted requirements, and PRD
        """
        conversation_history = interview_session.messages

        while len(conversation_history) < self.max_questions * 2:
            # 1. Retrieve relevant context from knowledge graph
            context = await self.context_retriever(
                ConversationContext(
                    user=interview_session.user,
                    messages=conversation_history[-6:],  # last 3 Q&A pairs
                    current_requirements=interview_session.requirements
                )
            )

            # 2. Generate next question based on context
            question = await self.question_generator(
                QuestionGenerationInput(
                    context=context,
                    conversation=conversation_history,
                    user=interview_session.user,
                    unanswered_areas=self._identify_gaps(interview_session)
                )
            )

            # Check if interview should conclude
            if question.is_concluding or self._should_conclude(interview_session):
                break

            # 3. Present question and get user response
            conversation_history.append(ChatMessage(
                role=ChatRole.ASSISTANT,
                content=question.text
            ))

            # This would pause for user input in actual implementation
            user_response = await self._get_user_response(question)
            conversation_history.append(ChatMessage(
                role=ChatRole.USER,
                content=user_response
            ))

            # 4. Analyze response to extract structured information
            analysis = await self.response_analyzer(
                ResponseAnalysisInput(
                    question=question,
                    response=user_response,
                    user=interview_session.user
                )
            )

            # 5. Update knowledge graph with new information
            knowledge_update = self._create_knowledge_update(
                user=interview_session.user,
                analysis=analysis,
                context=conversation_history
            )
            await self.knowledge_updater(knowledge_update)

            # 6. Track evolution: learn from this interaction
            await self.evolution_tracker.record_interaction(
                question=question,
                response=user_response,
                analysis=analysis,
                effectiveness=analysis.information_gain
            )

            # Update session state
            interview_session.requirements.extend(analysis.extracted_requirements)
            interview_session.use_cases.extend(analysis.extracted_use_cases)

        # 7. Generate final PRD
        prd = await self._generate_prd(interview_session)

        return InterviewResult(
            session=interview_session,
            prd=prd,
            conversation=conversation_history,
            confidence_score=self._calculate_confidence(interview_session)
        )
```

### 4. Self-Evolution Mechanism

The agent evolves through multiple mechanisms:

#### A. OMEGA Optimization (Built-in)
```python
# Training the interview agent to improve question quality
interview_agent_program = synalinks.Program(
    inputs=interview_session_input,
    outputs=interview_result_output
)

interview_agent_program.compile(
    optimizer=synalinks.optimizers.OMEGA(
        language_model=language_model,
        embedding_model=embedding_model
    ),
    reward=InterviewQualityReward(
        # Reward based on:
        # - Information completeness
        # - User engagement (conversation length)
        # - PRD quality
        # - Requirement discovery rate
    )
)

# Train on historical interviews
await interview_agent_program.fit(
    x=historical_sessions,
    y=expected_outcomes,
    epochs=10
)
```

#### B. Question Strategy Evolution
```python
class EvolutionTracker:
    """
    Tracks effectiveness of different questioning strategies
    and adapts the agent's approach over time.
    """

    async def record_interaction(
        self,
        question: InterviewQuestion,
        response: str,
        analysis: AnalyzedResponse,
        effectiveness: float
    ):
        """
        Store question strategy performance.
        Effectiveness metrics:
        - Information gain: new entities/relations discovered
        - Response depth: length and detail of answer
        - Follow-up potential: opens new discussion areas
        """
        await self.knowledge_base.add_entities([
            QuestionStrategy(
                label="QuestionStrategy",
                question_type=question.question_type,
                context_pattern=self._extract_pattern(question.context),
                effectiveness_score=effectiveness,
                timestamp=datetime.now().isoformat()
            )
        ])

    async def get_best_strategy(
        self,
        current_context: ConversationContext
    ) -> QuestionStrategy:
        """
        Retrieve most effective questioning strategy
        for similar contexts from past interactions.
        """
        # Vector similarity search on context patterns
        similar_strategies = await self.knowledge_base.query_similar(
            entity_type=QuestionStrategy,
            embedding=current_context.embedding,
            top_k=3
        )

        # Weight by recency and effectiveness
        return self._select_strategy(similar_strategies)
```

#### C. Graph-Based Learning
```python
class AdaptiveQuestionGenerator:
    """
    Uses graph patterns to identify effective question sequences.
    """

    async def generate_next_question(
        self,
        session: InterviewSession
    ) -> InterviewQuestion:
        # Find successful interview patterns from graph
        successful_paths = await self.knowledge_base.query("""
            MATCH (u:User)-[:HasRequirement]->(r:Requirement)
            WHERE u.industry = $user_industry
            WITH r, count(*) as frequency
            ORDER BY frequency DESC
            LIMIT 5
            RETURN r
        """, user_industry=session.user.industry)

        # Identify gaps in current session
        discovered = set(r.name for r in session.requirements)
        common_missing = [
            r for r in successful_paths
            if r.name not in discovered
        ]

        # Generate targeted question
        if common_missing:
            return await self._create_targeted_question(
                missing_requirement=common_missing[0],
                user=session.user
            )
        else:
            return await self._create_exploratory_question(session)
```

### 5. PRD Generation Module

```python
class PRDGenerator(synalinks.Module):
    """
    Generates comprehensive Product Requirements Document
    from interview knowledge graph.
    """

    def __init__(
        self,
        language_model: synalinks.LanguageModel,
        knowledge_base: synalinks.KnowledgeBase,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.language_model = language_model
        self.knowledge_base = knowledge_base

    async def call(self, session: InterviewSession) -> PRD:
        # 1. Retrieve all user-related knowledge
        user_graph = await self._retrieve_user_subgraph(session.user)

        # 2. Organize requirements by category and priority
        organized_reqs = self._organize_requirements(user_graph)

        # 3. Generate PRD sections
        executive_summary = await self._generate_section(
            "executive_summary",
            user_graph=user_graph,
            template=PRD_TEMPLATES["executive_summary"]
        )

        goals_objectives = await self._generate_section(
            "goals_objectives",
            user_graph=user_graph,
            template=PRD_TEMPLATES["goals_objectives"]
        )

        requirements_spec = await self._generate_section(
            "requirements",
            user_graph=user_graph,
            organized_reqs=organized_reqs,
            template=PRD_TEMPLATES["requirements"]
        )

        use_cases = await self._generate_section(
            "use_cases",
            user_graph=user_graph,
            template=PRD_TEMPLATES["use_cases"]
        )

        technical_spec = await self._generate_section(
            "technical_specification",
            user_graph=user_graph,
            template=PRD_TEMPLATES["technical_spec"]
        )

        # 4. Generate plugin recommendations
        plugin_matches = await self._match_plugins(organized_reqs)

        return PRD(
            user=session.user,
            executive_summary=executive_summary,
            goals_objectives=goals_objectives,
            requirements=requirements_spec,
            use_cases=use_cases,
            technical_specification=technical_spec,
            plugin_recommendations=plugin_matches,
            confidence_score=self._calculate_prd_confidence(user_graph),
            generated_at=datetime.now().isoformat()
        )

    async def _match_plugins(
        self,
        requirements: OrganizedRequirements
    ) -> List[PluginMatch]:
        """
        Match requirements to existing plugins using graph RAG.
        """
        matches = []

        for requirement in requirements.all_requirements:
            # Semantic search for matching plugins
            similar_plugins = await self.knowledge_base.query_similar(
                entity_type=Plugin,
                query_text=requirement.description,
                top_k=5
            )

            # Score each plugin
            for plugin in similar_plugins:
                score = await self._calculate_match_score(
                    requirement=requirement,
                    plugin=plugin
                )

                if score > 0.6:  # threshold
                    matches.append(PluginMatch(
                        requirement=requirement,
                        plugin=plugin,
                        score=score,
                        reasoning=await self._explain_match(
                            requirement, plugin, score
                        )
                    ))

                    # Update graph with match
                    await self.knowledge_base.add_relations([
                        PluginSatisfiesRequirement(
                            subj=plugin,
                            label="PluginSatisfiesRequirement",
                            obj=requirement,
                            match_score=score,
                            reasoning=matches[-1].reasoning
                        )
                    ])

        return sorted(matches, key=lambda m: m.score, reverse=True)
```

### 6. Complete Interview Flow (Functional API)

```python
async def create_interview_agent_program():
    """
    Assemble the complete interview agent using Synalinks Functional API.
    """
    # Initialize components
    lm = synalinks.LanguageModel(model="anthropic/claude-sonnet-4")
    embedding_model = synalinks.EmbeddingModel(model="openai/text-embedding-3-large")

    # Initialize knowledge base with interview schema
    kb = synalinks.KnowledgeBase(
        uri="neo4j://localhost:7687",
        entity_models=[User, Requirement, UseCase, Technology, Goal, Plugin],
        relation_models=[
            HasRequirement, HasUseCase, RequirementNeedsTechnology,
            UseCaseSupportsGoal, PluginSatisfiesRequirement, LeadsTo
        ],
        embedding_model=embedding_model
    )

    # Define inputs
    session_input = synalinks.Input(shape=(InterviewSession,))

    # Build interview pipeline
    interview_agent = InterviewAgent(
        language_model=lm,
        knowledge_base=kb,
        max_questions=20
    )

    # Execute interview
    interview_result = interview_agent(session_input)

    # Generate PRD
    prd_generator = PRDGenerator(
        language_model=lm,
        knowledge_base=kb
    )
    final_prd = prd_generator(interview_result)

    # Create program
    program = synalinks.Program(
        inputs=session_input,
        outputs=final_prd,
        name="InterviewAgentProgram"
    )

    # Compile with evolution optimizer
    program.compile(
        optimizer=synalinks.optimizers.OMEGA(
            language_model=lm,
            embedding_model=embedding_model
        ),
        reward=CompositeInterviewReward(
            completeness_weight=0.4,
            user_satisfaction_weight=0.3,
            prd_quality_weight=0.3
        )
    )

    return program
```

### 7. Plugin Marketplace Integration

```python
class PluginMarketplace:
    """
    Manages plugin discovery, recommendation, and installation
    based on user PRDs.
    """

    def __init__(
        self,
        knowledge_base: synalinks.KnowledgeBase,
        language_model: synalinks.LanguageModel
    ):
        self.kb = knowledge_base
        self.lm = language_model

    async def index_plugins(self, plugin_directory: str):
        """
        Index all available Claude Code plugins into knowledge graph.
        """
        plugins = await self._discover_plugins(plugin_directory)

        for plugin_meta in plugins:
            plugin = Plugin(
                label="Plugin",
                name=plugin_meta.name,
                description=plugin_meta.description,
                capabilities=plugin_meta.capabilities,
                categories=plugin_meta.categories,
                install_count=0,
                rating=0.0
            )

            await self.kb.add_entities([plugin])

    async def recommend_plugins(
        self,
        prd: PRD,
        top_k: int = 10
    ) -> List[PluginRecommendation]:
        """
        Recommend plugins based on PRD requirements.
        Uses hybrid search: semantic + graph-based.
        """
        recommendations = []

        # For each requirement in PRD
        for requirement in prd.requirements.all_requirements:
            # Semantic search
            semantic_matches = await self.kb.query_similar(
                entity_type=Plugin,
                query_text=f"{requirement.name}: {requirement.description}",
                top_k=5
            )

            # Graph-based search (find plugins used by similar users)
            graph_matches = await self.kb.query("""
                MATCH (u1:User)-[:HasRequirement]->(r1:Requirement)
                WHERE r1.category = $category
                MATCH (u2:User)-[:HasRequirement]->(r2:Requirement)
                WHERE r2.category = $category
                MATCH (p:Plugin)-[:PluginSatisfiesRequirement]->(r2)
                WHERE u1.id = $user_id AND u2.id <> $user_id
                WITH p, avg(p.rating) as avg_rating, count(*) as usage_count
                ORDER BY usage_count DESC, avg_rating DESC
                LIMIT 5
                RETURN p
            """, category=requirement.category, user_id=prd.user.id)

            # Combine and score
            all_matches = self._deduplicate_and_score(
                semantic_matches,
                graph_matches,
                requirement
            )

            recommendations.extend(all_matches)

        # Deduplicate across requirements and rank
        final_recs = self._rank_recommendations(recommendations, prd)

        return final_recs[:top_k]

    async def create_personalized_bundle(
        self,
        prd: PRD
    ) -> PluginBundle:
        """
        Create a curated bundle of plugins tailored to user's PRD.
        """
        recommendations = await self.recommend_plugins(prd, top_k=15)

        # Optimize bundle: maximize requirement coverage, minimize redundancy
        bundle = self._optimize_bundle(
            recommendations=recommendations,
            requirements=prd.requirements.all_requirements,
            max_plugins=10
        )

        # Generate installation script
        install_script = self._generate_install_script(bundle)

        # Generate configuration guide
        config_guide = await self._generate_config_guide(
            bundle=bundle,
            user=prd.user,
            requirements=prd.requirements
        )

        return PluginBundle(
            plugins=bundle,
            install_script=install_script,
            configuration_guide=config_guide,
            estimated_setup_time=self._estimate_setup_time(bundle)
        )
```

## Implementation Phases

### Phase 1: Foundation (Week 1-2)
- [ ] Define data models for interview entities and relations
- [ ] Set up knowledge base with Neo4j/MemGraph
- [ ] Create basic InterviewAgent module structure
- [ ] Implement conversation state management

### Phase 2: Core Interview Logic (Week 2-3)
- [ ] Implement question generation with context retrieval
- [ ] Build response analysis pipeline
- [ ] Create knowledge graph update mechanism
- [ ] Add conversation flow control

### Phase 3: Self-Evolution (Week 3-4)
- [ ] Implement EvolutionTracker
- [ ] Create question strategy learning
- [ ] Integrate OMEGA optimizer
- [ ] Add effectiveness metrics

### Phase 4: PRD Generation (Week 4-5)
- [ ] Build PRDGenerator module
- [ ] Create PRD templates
- [ ] Implement requirement organization
- [ ] Add confidence scoring

### Phase 5: Plugin Marketplace (Week 5-6)
- [ ] Index existing Claude Code plugins
- [ ] Implement plugin recommendation engine
- [ ] Create bundle optimization
- [ ] Build installation automation

### Phase 6: Integration & Testing (Week 6-7)
- [ ] End-to-end testing
- [ ] User experience refinement
- [ ] Performance optimization
- [ ] Documentation

## Key Design Decisions

### 1. Why Graph RAG?
- **Relational context**: User requirements are interconnected
- **Pattern discovery**: Learn from successful interview paths
- **Explainability**: Can trace why questions were asked
- **Flexibility**: Easy to add new entity/relation types

### 2. Why Self-Evolution?
- **Continuous improvement**: Agent gets better with each interview
- **Personalization**: Adapts to different user types and industries
- **Efficiency**: Learns optimal question sequences
- **Quality**: Improves PRD completeness over time

### 3. Why Synalinks?
- **Native graph support**: Built-in knowledge base integration
- **Modular composition**: Easy to build complex agent workflows
- **Training ready**: OMEGA optimizer for prompt optimization
- **Serializable**: Can save/load agent state and learning

## Example Usage

```python
import synalinks

# Create interview program
program = await create_interview_agent_program()

# Start new interview session
session = InterviewSession(
    user=User(
        label="User",
        name="Jane Doe",
        role="Software Engineer",
        experience_level="intermediate",
        industry="SaaS"
    ),
    messages=[],
    requirements=[],
    use_cases=[]
)

# Run interview
result = await program(session)

print(f"Generated PRD with {len(result.prd.requirements)} requirements")
print(f"Confidence score: {result.prd.confidence_score:.2f}")
print(f"Recommended plugins: {len(result.prd.plugin_recommendations)}")

# Get plugin bundle
marketplace = PluginMarketplace(kb, lm)
bundle = await marketplace.create_personalized_bundle(result.prd)

print(f"Installing {len(bundle.plugins)} plugins...")
await bundle.install()
```

## Future Enhancements

1. **Multi-modal interviews**: Support image/file uploads during conversation
2. **Real-time collaboration**: Multiple stakeholders in interview
3. **Active learning**: Agent requests human feedback on uncertain requirements
4. **A/B testing**: Compare different interview strategies
5. **Integration with Claude Code API**: Direct plugin installation and configuration
6. **Feedback loop**: Track plugin usage to improve future recommendations
