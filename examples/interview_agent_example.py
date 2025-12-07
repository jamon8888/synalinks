"""
Interview Agent Example

Demonstrates how to use the Interview Agent system to:
1. Conduct an intelligent onboarding interview
2. Generate a comprehensive PRD
3. Recommend personalized Claude Code plugins
4. Enable self-evolution through learning

This example shows both programmatic usage and integration patterns.
"""

import asyncio
from datetime import datetime
from typing import Optional

import synalinks
from synalinks.src.modules.interview_agent import (
    InterviewAgent,
    PRDGenerator,
    PluginMarketplace,
    EvolutionTracker,
    User,
    InterviewSession,
    Requirement,
    UseCase,
    Technology,
    Goal,
    Plugin,
)


# ============================================================================
# EXAMPLE 1: Basic Interview Flow
# ============================================================================


async def example_basic_interview():
    """
    Basic interview agent usage with simulated user responses.
    """
    print("=" * 70)
    print("EXAMPLE 1: Basic Interview Flow")
    print("=" * 70)

    # 1. Setup
    lm = synalinks.LanguageModel(model="anthropic/claude-sonnet-4")
    embedding_model = synalinks.EmbeddingModel(model="openai/text-embedding-3-large")

    kb = synalinks.KnowledgeBase(
        uri="neo4j://localhost:7687",
        entity_models=[User, Requirement, UseCase, Technology, Goal, Plugin],
        embedding_model=embedding_model,
    )

    # 2. Create interview agent
    interview_agent = InterviewAgent(
        language_model=lm,
        knowledge_base=kb,
        max_questions=10,
        user_response_callback=None,  # Will use test data
    )

    # 3. Create user and session
    user = User(
        label="User",
        name="Alice Johnson",
        role="Senior Software Engineer",
        experience_level="advanced",
        industry="FinTech",
        team_size=12,
    )

    session = InterviewSession(
        user=user,
        messages=[],
        requirements=[],
        use_cases=[],
        technologies=[],
        goals=[],
    )

    # 4. Build and run interview
    await interview_agent.build(input_shape=(InterviewSession,))
    result = await interview_agent(session)

    # 5. Display results
    print(f"\n✓ Interview completed!")
    print(f"  - Questions asked: {result.questions_asked}")
    print(f"  - Requirements discovered: {len(result.session.requirements)}")
    print(f"  - Use cases identified: {len(result.session.use_cases)}")
    print(f"  - Technologies mentioned: {len(result.session.technologies)}")
    print(f"  - Goals identified: {len(result.session.goals)}")
    print(f"  - Confidence score: {result.confidence_score:.0%}")

    return result


# ============================================================================
# EXAMPLE 2: Generate PRD from Interview
# ============================================================================


async def example_generate_prd(interview_result):
    """
    Generate a comprehensive PRD from interview results.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: PRD Generation")
    print("=" * 70)

    # 1. Setup PRD generator
    lm = synalinks.LanguageModel(model="anthropic/claude-sonnet-4")
    kb = synalinks.KnowledgeBase(
        uri="neo4j://localhost:7687",
        entity_models=[User, Requirement, UseCase, Technology, Goal, Plugin],
    )

    prd_generator = PRDGenerator(
        language_model=lm,
        knowledge_base=kb,
    )

    # 2. Build and generate PRD
    await prd_generator.build(input_shape=(InterviewSession,))
    prd = await prd_generator(interview_result.session)

    # 3. Display PRD
    print(f"\n✓ PRD generated!")
    print(f"  - Sections: {len(prd.sections)}")
    print(f"  - Total requirements: {len(prd.requirements.all_requirements)}")
    print(f"  - Requirement categories: {len(prd.requirements.by_category)}")
    print(f"  - Confidence score: {prd.confidence_score:.0%}")

    print("\n--- Executive Summary ---")
    print(prd.executive_summary[:300] + "...")

    print("\n--- Requirements by Category ---")
    for category, reqs in prd.requirements.by_category.items():
        print(f"  {category}: {len(reqs)} requirements")

    # 4. Export as Markdown
    markdown_prd = prd_generator.format_prd_markdown(prd)
    with open("prd_output.md", "w") as f:
        f.write(markdown_prd)

    print(f"\n✓ PRD saved to prd_output.md")

    return prd


# ============================================================================
# EXAMPLE 3: Plugin Recommendations
# ============================================================================


async def example_plugin_recommendations(prd):
    """
    Recommend plugins based on PRD and create personalized bundle.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Plugin Recommendations")
    print("=" * 70)

    # 1. Setup marketplace
    lm = synalinks.LanguageModel(model="anthropic/claude-sonnet-4")
    kb = synalinks.KnowledgeBase(uri="neo4j://localhost:7687")

    marketplace = PluginMarketplace(
        knowledge_base=kb,
        language_model=lm,
    )

    # 2. Index example plugins
    # In production, this would scan a real plugin directory
    example_plugins = [
        Plugin(
            label="Plugin",
            name="code-reviewer",
            description="Automated code review with best practices",
            capabilities=["code_analysis", "review", "suggestions"],
            categories=["coding", "testing"],
            rating=4.5,
            install_count=1250,
        ),
        Plugin(
            label="Plugin",
            name="test-generator",
            description="Generate unit tests from code",
            capabilities=["testing", "test_generation", "coverage"],
            categories=["testing"],
            rating=4.2,
            install_count=890,
        ),
        Plugin(
            label="Plugin",
            name="doc-writer",
            description="Auto-generate documentation",
            capabilities=["documentation", "markdown", "api_docs"],
            categories=["documentation"],
            rating=4.0,
            install_count=650,
        ),
    ]

    for plugin in example_plugins:
        marketplace.indexed_plugins[plugin.name] = plugin

    # 3. Get recommendations
    recommendations = await marketplace.recommend_plugins(prd, top_k=10)

    print(f"\n✓ Found {len(recommendations)} plugin recommendations")

    for i, rec in enumerate(recommendations[:5], 1):
        print(f"\n{i}. {rec.plugin.name} (Score: {rec.relevance_score:.0%})")
        print(f"   {rec.plugin.description}")
        print(f"   Priority: {rec.installation_priority}/10")
        print(
            f"   Satisfies: {', '.join(r.name for r in rec.matched_requirements[:2])}"
        )

    # 4. Create personalized bundle
    bundle = await marketplace.create_personalized_bundle(prd, max_plugins=5)

    print(f"\n✓ Created personalized bundle")
    print(f"  - Plugins: {len(bundle.plugins)}")
    print(f"  - Requirement coverage: {bundle.coverage_score:.0%}")
    print(f"  - Estimated setup time: {bundle.estimated_setup_time}")

    # 5. Save installation script
    with open("install_plugins.sh", "w") as f:
        f.write(bundle.install_script)

    with open("plugin_config_guide.md", "w") as f:
        f.write(bundle.configuration_guide)

    print("\n✓ Installation script saved to install_plugins.sh")
    print("✓ Configuration guide saved to plugin_config_guide.md")

    return bundle


# ============================================================================
# EXAMPLE 4: Self-Evolution and Learning
# ============================================================================


async def example_evolution_tracking():
    """
    Demonstrate self-evolution capabilities.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Self-Evolution and Learning")
    print("=" * 70)

    # 1. Setup evolution tracker
    kb = synalinks.KnowledgeBase(uri="neo4j://localhost:7687")
    tracker = EvolutionTracker(knowledge_base=kb)

    # 2. Simulate recording interactions
    from synalinks.src.modules.interview_agent import InterviewQuestion, AnalyzedResponse

    print("\nRecording interview interactions...")

    # Simulate 5 interactions
    for i in range(5):
        question = InterviewQuestion(
            text=f"Sample question {i}",
            question_type=["open_ended", "specific", "follow_up"][i % 3],
            purpose="Test interaction",
            is_concluding=False,
        )

        analysis = AnalyzedResponse(
            extracted_requirements=[],
            information_gain=0.6 + (i * 0.05),  # Increasing over time
            confidence=0.7,
            sentiment="positive",
        )

        await tracker.record_interaction(
            question=question,
            response="Sample response",
            analysis=analysis,
            effectiveness=0.6 + (i * 0.05),
            context={"session_id": "test_session", "phase": "exploration"},
        )

    # 3. Analyze learning
    insights = await tracker.get_learning_insights()

    print(f"\n✓ Learning Insights:")
    print(f"  - Total interactions: {insights['total_interactions']}")
    print(f"  - Average effectiveness: {insights['avg_effectiveness']:.2%}")
    print(f"  - Average info gain: {insights['avg_info_gain']:.2%}")
    print(
        f"  - Most effective question type: {insights['most_effective_question_type']}"
    )

    print("\n  Question type performance:")
    for qtype, score in insights["question_type_performance"].items():
        print(f"    - {qtype}: {score:.2%}")

    # 4. Get session performance
    session_perf = await tracker.analyze_session_performance("test_session")
    print(f"\n✓ Session Performance:")
    print(f"  - Total interactions: {session_perf['total_interactions']}")
    print(f"  - Average effectiveness: {session_perf['avg_effectiveness']:.2%}")
    print(f"  - Question types used: {session_perf['question_type_distribution']}")

    return tracker


# ============================================================================
# EXAMPLE 5: Complete End-to-End Flow
# ============================================================================


async def example_complete_flow():
    """
    Complete end-to-end example: Interview → PRD → Plugin Recommendations.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Complete End-to-End Flow")
    print("=" * 70)

    # 1. Mock user for testing
    user = User(
        label="User",
        name="Bob Developer",
        role="Full Stack Developer",
        experience_level="intermediate",
        industry="E-commerce",
        team_size=5,
    )

    # Pre-populate with some mock data for quick demo
    session = InterviewSession(
        user=user,
        requirements=[
            Requirement(
                label="Requirement",
                name="Automated Testing",
                description="Need automated unit and integration test generation",
                priority=5,
                category="testing",
                confidence_score=0.9,
                discovered_at=datetime.now().isoformat(),
            ),
            Requirement(
                label="Requirement",
                name="Code Review",
                description="Automated code review for pull requests",
                priority=4,
                category="coding",
                confidence_score=0.85,
                discovered_at=datetime.now().isoformat(),
            ),
        ],
        use_cases=[
            UseCase(
                label="UseCase",
                name="PR Review Workflow",
                description="Review pull requests for code quality and best practices",
                frequency="daily",
                pain_points=["Manual review takes too long", "Inconsistent standards"],
            )
        ],
        technologies=[
            Technology(label="Technology", name="Python", type="language", proficiency="expert"),
            Technology(label="Technology", name="React", type="framework", proficiency="intermediate"),
        ],
        goals=[
            Goal(
                label="Goal",
                name="Improve Code Quality",
                description="Reduce bugs and improve maintainability",
                importance=5,
                success_criteria=["30% reduction in bugs", "Faster PR reviews"],
            )
        ],
        questions_asked=8,
    )

    print("\n1. Interview Session (using mock data)")
    print(f"   User: {user.name} ({user.role})")
    print(f"   Requirements: {len(session.requirements)}")
    print(f"   Use cases: {len(session.use_cases)}")

    # 2. Generate PRD
    print("\n2. Generating PRD...")
    lm = synalinks.LanguageModel(model="anthropic/claude-sonnet-4")
    kb = synalinks.KnowledgeBase(uri="neo4j://localhost:7687")

    prd_gen = PRDGenerator(language_model=lm, knowledge_base=kb)
    await prd_gen.build(input_shape=(InterviewSession,))
    prd = await prd_gen(session)

    print(f"   ✓ PRD generated with {len(prd.sections)} sections")
    print(f"   Confidence: {prd.confidence_score:.0%}")

    # 3. Recommend plugins
    print("\n3. Recommending plugins...")
    marketplace = PluginMarketplace(knowledge_base=kb, language_model=lm)

    # Add sample plugins
    marketplace.indexed_plugins["test-gen"] = Plugin(
        label="Plugin",
        name="test-generator",
        description="Generate tests automatically",
        capabilities=["testing", "test_generation"],
        categories=["testing"],
        rating=4.5,
    )

    recommendations = await marketplace.recommend_plugins(prd, top_k=5)
    print(f"   ✓ Found {len(recommendations)} recommendations")

    # 4. Create bundle
    print("\n4. Creating plugin bundle...")
    bundle = await marketplace.create_personalized_bundle(prd)
    print(f"   ✓ Bundle with {len(bundle.plugins)} plugins")
    print(f"   Coverage: {bundle.coverage_score:.0%}")
    print(f"   Setup time: {bundle.estimated_setup_time}")

    print("\n" + "=" * 70)
    print("✓ Complete flow finished successfully!")
    print("=" * 70)

    return {
        "session": session,
        "prd": prd,
        "recommendations": recommendations,
        "bundle": bundle,
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================


async def main():
    """
    Run all examples.

    Note: Some examples require a running Neo4j/MemGraph instance.
    Set up with: docker run -p 7687:7687 neo4j:latest
    """
    print("\n" + "=" * 70)
    print("INTERVIEW AGENT EXAMPLES")
    print("=" * 70)
    print("\nThese examples demonstrate the Interview Agent system.")
    print("Some features require a graph database (Neo4j/MemGraph).")
    print()

    try:
        # Run complete flow example (works without KB)
        result = await example_complete_flow()

        # Optionally run other examples
        # interview_result = await example_basic_interview()
        # prd = await example_generate_prd(interview_result)
        # bundle = await example_plugin_recommendations(prd)
        # tracker = await example_evolution_tracking()

        print("\n✅ All examples completed successfully!")

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
