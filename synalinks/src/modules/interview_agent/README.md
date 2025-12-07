# Interview Agent - Self-Evolving Onboarding System

A sophisticated interview agent system built with Synalinks that conducts intelligent onboarding conversations using Graph RAG to understand individual AI needs and generate comprehensive Product Requirements Documents (PRDs).

## Features

### 🎯 Core Capabilities

- **Intelligent Interviewing**: Conducts context-aware, multi-turn conversations that adapt based on user responses
- **Graph RAG Integration**: Leverages knowledge graph for context retrieval and pattern learning
- **PRD Generation**: Automatically generates comprehensive Product Requirements Documents
- **Plugin Marketplace**: Recommends and bundles Claude Code plugins based on discovered needs
- **Self-Evolution**: Learns from past interactions to improve questioning strategies over time

### 🧠 Self-Learning Mechanisms

1. **Question Strategy Tracking**: Records effectiveness of different question types and patterns
2. **Pattern Discovery**: Identifies successful interview paths from similar users
3. **Adaptive Generation**: Adjusts questioning approach based on learned patterns
4. **OMEGA Optimization**: Uses in-context RL to optimize prompts and instructions

## Architecture

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

## Quick Start

### Installation

```bash
# Install synalinks
pip install synalinks

# Install graph database (choose one)
# Neo4j
docker run -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest

# Or MemGraph
docker run -p 7687:7687 memgraph/memgraph
```

### Basic Usage

```python
import asyncio
import synalinks
from synalinks.src.modules.interview_agent import (
    InterviewAgent,
    PRDGenerator,
    PluginMarketplace,
    User,
    InterviewSession,
)

async def run_interview():
    # 1. Setup models and knowledge base
    lm = synalinks.LanguageModel(model="anthropic/claude-sonnet-4")
    emb = synalinks.EmbeddingModel(model="openai/text-embedding-3-large")

    kb = synalinks.KnowledgeBase(
        uri="neo4j://localhost:7687",
        embedding_model=emb,
    )

    # 2. Create interview agent
    agent = InterviewAgent(
        language_model=lm,
        knowledge_base=kb,
        max_questions=20,
    )

    # 3. Create user and session
    user = User(
        label="User",
        name="Jane Doe",
        role="Software Engineer",
        experience_level="intermediate",
        industry="SaaS",
    )

    session = InterviewSession(user=user)

    # 4. Run interview
    await agent.build(input_shape=(InterviewSession,))
    result = await agent(session)

    # 5. Generate PRD
    prd_gen = PRDGenerator(language_model=lm, knowledge_base=kb)
    await prd_gen.build(input_shape=(InterviewSession,))
    prd = await prd_gen(result.session)

    # 6. Get plugin recommendations
    marketplace = PluginMarketplace(knowledge_base=kb, language_model=lm)
    recommendations = await marketplace.recommend_plugins(prd, top_k=10)

    # 7. Create plugin bundle
    bundle = await marketplace.create_personalized_bundle(prd)

    print(f"Interview complete!")
    print(f"Requirements: {len(prd.requirements.all_requirements)}")
    print(f"Recommended plugins: {len(recommendations)}")
    print(f"Bundle coverage: {bundle.coverage_score:.0%}")

    return prd, bundle

# Run
asyncio.run(run_interview())
```

### With User Response Callback

For interactive interviews, provide a callback function:

```python
async def get_user_input(question):
    """Custom callback to get real user responses."""
    print(f"\nAgent: {question.text}")
    response = input("You: ")
    return response

agent = InterviewAgent(
    language_model=lm,
    knowledge_base=kb,
    user_response_callback=get_user_input,
)
```

## Components

### 1. InterviewAgent

The main agent that conducts interviews.

**Key Features:**
- Multi-phase interview flow (introduction → exploration → deep_dive → validation)
- Dynamic question generation based on context
- Automatic knowledge graph updates
- Gap identification to ensure comprehensive coverage

**Example:**
```python
agent = InterviewAgent(
    language_model=lm,
    knowledge_base=kb,
    max_questions=20,
    adaptation_threshold=0.7,
)

result = await agent(session)
```

### 2. PRDGenerator

Generates structured Product Requirements Documents.

**PRD Sections:**
- Executive Summary
- Goals & Objectives
- Requirements (organized by category and priority)
- Use Cases
- Technical Specification
- Plugin Recommendations

**Example:**
```python
prd_gen = PRDGenerator(language_model=lm, knowledge_base=kb)
prd = await prd_gen(session)

# Export as Markdown
markdown = prd_gen.format_prd_markdown(prd)
with open("prd.md", "w") as f:
    f.write(markdown)
```

### 3. EvolutionTracker

Enables self-learning and strategy improvement.

**Tracks:**
- Question effectiveness (information gain)
- Successful question patterns
- Optimal question sequences for user types
- Performance trends

**Example:**
```python
tracker = EvolutionTracker(knowledge_base=kb)

# Record interaction
await tracker.record_interaction(
    question=question,
    response=response,
    analysis=analysis,
    effectiveness=0.85,
)

# Get insights
insights = await tracker.get_learning_insights()
print(f"Most effective question type: {insights['most_effective_question_type']}")
```

### 4. PluginMarketplace

Recommends and bundles Claude Code plugins.

**Features:**
- Hybrid matching (semantic + graph-based + category)
- Bundle optimization (maximize coverage, minimize redundancy)
- Installation script generation
- Configuration guide creation

**Example:**
```python
marketplace = PluginMarketplace(kb=kb, lm=lm)

# Index plugins
await marketplace.index_plugins("./plugins")

# Get recommendations
recs = await marketplace.recommend_plugins(prd, top_k=10)

# Create bundle
bundle = await marketplace.create_personalized_bundle(prd)

# Save installation script
with open("install.sh", "w") as f:
    f.write(bundle.install_script)
```

## Data Models

### Knowledge Graph Schema

**Entities:**
- `User`: Individual being interviewed
- `Requirement`: Specific AI/automation need
- `UseCase`: Workflow or scenario
- `Technology`: Tool, language, or framework
- `Goal`: Objective to achieve
- `Plugin`: Claude Code plugin
- `QuestionStrategy`: Learned questioning pattern

**Relations:**
- `HasRequirement`: User has a requirement
- `HasUseCase`: User has a use case
- `RequirementNeedsTechnology`: Requirement depends on technology
- `PluginSatisfiesRequirement`: Plugin addresses requirement
- `UseCaseSupportsGoal`: Use case helps achieve goal
- `LeadsTo`: Requirement relationships (prerequisite, enhancement, etc.)

### Example Data Model Usage

```python
from synalinks.src.modules.interview_agent import (
    User, Requirement, UseCase, Technology, Goal,
    HasRequirement, PluginSatisfiesRequirement
)

# Create entities
user = User(
    label="User",
    name="Alice",
    role="Data Scientist",
    experience_level="advanced",
    industry="Healthcare",
)

requirement = Requirement(
    label="Requirement",
    name="Data Pipeline Automation",
    description="Automated ETL pipeline for medical data",
    priority=5,
    category="data_analysis",
    confidence_score=0.9,
    discovered_at="2025-01-15T10:30:00",
)

# Create relation
relation = HasRequirement(
    subj=user,
    label="HasRequirement",
    obj=requirement,
    discovered_at="2025-01-15T10:30:00",
    context="User mentioned spending 10 hours/week on manual ETL",
)

# Add to knowledge base
await kb.add_entities([user, requirement])
await kb.add_relations([relation])
```

## Training & Optimization

The Interview Agent can be optimized using Synalinks' OMEGA optimizer:

```python
# Create program
program = synalinks.Program(
    inputs=session_input,
    outputs=interview_result,
)

# Define reward function
from synalinks.rewards import LMAsJudge

reward = LMAsJudge(
    language_model=lm,
    criteria=[
        "Information completeness: Did the interview gather comprehensive requirements?",
        "User engagement: Was the conversation natural and productive?",
        "Requirement quality: Are discovered requirements clear and actionable?",
    ]
)

# Compile with OMEGA
program.compile(
    optimizer=synalinks.optimizers.OMEGA(
        language_model=lm,
        embedding_model=emb,
    ),
    reward=reward,
)

# Train on historical interviews
await program.fit(
    x=historical_sessions,
    y=expected_prds,
    epochs=10,
    batch_size=1,
)
```

## Advanced Usage

### Custom Question Strategy

```python
from synalinks.src.modules.interview_agent import AdaptiveQuestionGenerator

adaptive_gen = AdaptiveQuestionGenerator(
    language_model=lm,
    knowledge_base=kb,
    evolution_tracker=tracker,
)

# Generate optimized question based on learning
question = await adaptive_gen.generate_adaptive_question(
    session=session,
    conversation_history=messages,
    unanswered_areas=gaps,
)
```

### Plugin Bundle Optimization

```python
# Create custom bundle with specific constraints
bundle = await marketplace.create_personalized_bundle(
    prd=prd,
    max_plugins=5,  # Limit bundle size
)

# Calculate coverage
coverage = bundle.coverage_score
print(f"This bundle covers {coverage:.0%} of requirements")

# Get setup time estimate
print(f"Estimated setup time: {bundle.estimated_setup_time}")
```

### Export and Serialization

```python
# Export PRD as JSON
import json

prd_dict = prd.model_dump()
with open("prd.json", "w") as f:
    json.dump(prd_dict, f, indent=2)

# Export knowledge graph
# (Use Cypher export or Neo4j dump tools)
```

## Examples

See `examples/interview_agent_example.py` for comprehensive examples including:

1. Basic interview flow
2. PRD generation
3. Plugin recommendations
4. Self-evolution tracking
5. Complete end-to-end workflow

Run examples:
```bash
python examples/interview_agent_example.py
```

## Configuration

### Environment Variables

```bash
# Knowledge Base
export NEO4J_URI="neo4j://localhost:7687"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="password"

# Language Models (via LiteLLM)
export ANTHROPIC_API_KEY="sk-..."
export OPENAI_API_KEY="sk-..."

# Optional: Plugin directory
export PLUGIN_DIRECTORY="./plugins"
```

### Interview Settings

```python
agent = InterviewAgent(
    language_model=lm,
    knowledge_base=kb,
    max_questions=20,           # Maximum questions per interview
    adaptation_threshold=0.7,    # When to adapt strategy
)
```

## Integration

### With Anthropic Interviewer Dataset

```python
import requests

# Fetch dataset
response = requests.get(
    "https://datasets-server.huggingface.co/rows",
    params={
        "dataset": "Anthropic/AnthropicInterviewer",
        "config": "AnthropicInterviewer",
        "split": "workforce",
        "offset": 0,
        "length": 100,
    }
)

interview_data = response.json()

# Use for training/examples
# ... process and use data for agent training
```

### With MCP (Model Context Protocol)

```python
from synalinks.utils.mcp import MultiServerMCPClient

# Connect to MCP servers for additional tools
mcp_client = MultiServerMCPClient({
    "research": {
        "url": "http://localhost:8183/mcp/",
        "transport": "streamable_http",
    }
})

tools = await mcp_client.get_tools()

# Integrate tools with interview agent
# (e.g., for real-time web research during interview)
```

## Performance

### Optimization Tips

1. **Use caching**: Enable knowledge base caching for faster retrieval
2. **Batch operations**: Process multiple sessions in parallel
3. **Model selection**: Use `haiku` for quick operations, `sonnet` for complex reasoning
4. **Limit questions**: Set appropriate `max_questions` for your use case

### Benchmarks

On typical interviews (15-20 questions):
- Interview time: ~5-10 minutes (interactive)
- PRD generation: ~30-60 seconds
- Plugin recommendations: ~10-20 seconds
- Total end-to-end: ~10-15 minutes

## Roadmap

- [ ] Multi-modal support (images, file uploads)
- [ ] Real-time collaboration (multiple stakeholders)
- [ ] Active learning (agent requests clarification)
- [ ] A/B testing framework
- [ ] Direct Claude Code API integration
- [ ] Feedback loop from plugin usage

## Contributing

Contributions welcome! Areas of interest:
- Additional question strategies
- New plugin matching algorithms
- PRD templates for specific domains
- Training datasets
- Performance optimizations

## License

See main Synalinks license.

## Citation

If you use this Interview Agent system in your research, please cite:

```bibtex
@software{synalinks_interview_agent,
  title = {Interview Agent: Self-Evolving Onboarding System},
  author = {Synalinks Contributors},
  year = {2025},
  url = {https://github.com/synalinks/synalinks}
}
```

## Support

For issues, questions, or contributions:
- GitHub Issues: [synalinks/issues](https://github.com/synalinks/synalinks/issues)
- Documentation: [docs/interview_agent_architecture.md](../../../docs/interview_agent_architecture.md)
- Examples: [examples/interview_agent_example.py](../../../examples/interview_agent_example.py)
