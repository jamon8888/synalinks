"""
Use Trained Interview Agent

Load and use a trained interview agent for conducting new interviews.
"""

import asyncio
import synalinks
from synalinks.src.modules.interview_agent import (
    User,
    InterviewSession,
    PRDGenerator,
    PluginMarketplace,
)


async def interactive_interview():
    """
    Conduct an interactive interview using the trained agent.
    """
    print("=" * 70)
    print("INTERACTIVE INTERVIEW WITH TRAINED AGENT")
    print("=" * 70)

    # 1. Load trained model
    print("\n1. Loading trained model...")
    try:
        program = synalinks.Program.load("trained_interview_agent.json")
        program.load_variables("trained_interview_agent_vars.json")
        print("✓ Loaded trained model")
    except FileNotFoundError:
        print("⚠ Trained model not found. Using base model instead.")
        print("  Run train_interview_agent.py first to train the model.")

        # Fall back to base model
        lm = synalinks.LanguageModel(model="anthropic/claude-sonnet-4")
        kb = synalinks.KnowledgeBase(uri="neo4j://localhost:7687")

        from synalinks.src.modules.interview_agent import InterviewAgent

        agent = InterviewAgent(language_model=lm, knowledge_base=kb)
        await agent.build(input_shape=(InterviewSession,))

        session_input = synalinks.Input(shape=(InterviewSession,))
        result_output = agent(session_input)
        program = synalinks.Program(inputs=session_input, outputs=result_output)

    # 2. Get user information
    print("\n2. User Information")
    print("-" * 70)
    name = input("Your name: ").strip() or "User"
    role = input("Your role (e.g., Software Engineer): ").strip() or "Professional"
    industry = input("Your industry (e.g., Tech, Healthcare): ").strip() or "Technology"
    experience = input("Experience level (beginner/intermediate/advanced): ").strip() or "intermediate"

    user = User(
        label="User",
        name=name,
        role=role,
        experience_level=experience,
        industry=industry,
    )

    # 3. Create session
    session = InterviewSession(user=user)

    print(f"\n3. Starting Interview with {user.name}")
    print("-" * 70)
    print("The agent will ask you questions to understand your AI automation needs.")
    print("Answer naturally - the agent learns from your responses.")
    print("Type 'skip' to skip a question, or 'done' to finish early.")
    print()

    # 4. Run interview
    try:
        result = await program(session)

        print("\n" + "=" * 70)
        print("INTERVIEW COMPLETE")
        print("=" * 70)
        print(f"\n✓ Questions asked: {result.questions_asked}")
        print(f"✓ Requirements discovered: {len(result.session.requirements)}")
        print(f"✓ Use cases identified: {len(result.session.use_cases)}")
        print(f"✓ Confidence score: {result.confidence_score:.0%}")

        # 5. Show discovered requirements
        if result.session.requirements:
            print(f"\n📋 Discovered Requirements:")
            for i, req in enumerate(result.session.requirements[:5], 1):
                print(f"\n{i}. {req.name} (Priority: {req.priority}/5)")
                print(f"   {req.description}")
                print(f"   Category: {req.category}")

        # 6. Generate PRD
        print("\n5. Generating PRD...")
        lm = synalinks.LanguageModel(model="anthropic/claude-sonnet-4")
        kb = synalinks.KnowledgeBase(uri="neo4j://localhost:7687")

        prd_gen = PRDGenerator(language_model=lm, knowledge_base=kb)
        await prd_gen.build(input_shape=(InterviewSession,))
        prd = await prd_gen(result.session)

        # Save PRD
        markdown = prd_gen.format_prd_markdown(prd)
        filename = f"prd_{user.name.replace(' ', '_').lower()}.md"
        with open(filename, "w") as f:
            f.write(markdown)

        print(f"✓ PRD saved to {filename}")

        # 7. Plugin recommendations
        print("\n6. Finding plugin recommendations...")
        marketplace = PluginMarketplace(knowledge_base=kb, language_model=lm)

        # For demo, add some sample plugins
        from synalinks.src.modules.interview_agent import Plugin

        sample_plugins = [
            Plugin(
                label="Plugin",
                name="code-reviewer",
                description="Automated code review with best practices",
                capabilities=["code_analysis", "review", "suggestions", "best_practices"],
                categories=["coding", "testing"],
                rating=4.5,
            ),
            Plugin(
                label="Plugin",
                name="test-generator",
                description="Generate comprehensive unit tests",
                capabilities=["testing", "test_generation", "coverage", "mocking"],
                categories=["testing", "coding"],
                rating=4.3,
            ),
            Plugin(
                label="Plugin",
                name="doc-writer",
                description="Auto-generate documentation from code",
                capabilities=["documentation", "markdown", "api_docs", "comments"],
                categories=["documentation", "coding"],
                rating=4.0,
            ),
        ]

        for plugin in sample_plugins:
            marketplace.indexed_plugins[plugin.name] = plugin

        recommendations = await marketplace.recommend_plugins(prd, top_k=5)

        if recommendations:
            print(f"\n🔌 Recommended Plugins:")
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"\n{i}. {rec.plugin.name} (Match: {rec.relevance_score:.0%})")
                print(f"   {rec.plugin.description}")
                print(f"   Rating: {'★' * int(rec.plugin.rating)}{'☆' * (5-int(rec.plugin.rating))}")

        # 8. Create bundle
        bundle = await marketplace.create_personalized_bundle(prd, max_plugins=5)

        with open("install_plugins.sh", "w") as f:
            f.write(bundle.install_script)

        print(f"\n✓ Installation script: install_plugins.sh")
        print(f"  Plugins in bundle: {len(bundle.plugins)}")
        print(f"  Requirement coverage: {bundle.coverage_score:.0%}")
        print(f"  Estimated setup time: {bundle.estimated_setup_time}")

        print("\n" + "=" * 70)
        print("✅ INTERVIEW SESSION COMPLETE")
        print("=" * 70)
        print(f"\nGenerated files:")
        print(f"  - {filename} (PRD)")
        print(f"  - install_plugins.sh (Plugin installation)")
        print(f"\nNext steps:")
        print(f"  1. Review the PRD")
        print(f"  2. Install recommended plugins")
        print(f"  3. Customize based on your workflow")

    except Exception as e:
        print(f"\n❌ Error during interview: {e}")
        import traceback
        traceback.print_exc()


async def batch_interviews():
    """
    Run multiple interviews in batch mode (for testing/evaluation).
    """
    print("=" * 70)
    print("BATCH INTERVIEW MODE")
    print("=" * 70)

    # Load trained model
    try:
        program = synalinks.Program.load("trained_interview_agent.json")
        program.load_variables("trained_interview_agent_vars.json")
        print("✓ Loaded trained model\n")
    except FileNotFoundError:
        print("❌ Trained model not found. Please train first.")
        return

    # Test users
    test_users = [
        User(
            label="User",
            name="Alice",
            role="Data Scientist",
            experience_level="advanced",
            industry="Healthcare",
        ),
        User(
            label="User",
            name="Bob",
            role="Frontend Developer",
            experience_level="intermediate",
            industry="E-commerce",
        ),
        User(
            label="User",
            name="Carol",
            role="DevOps Engineer",
            experience_level="expert",
            industry="FinTech",
        ),
    ]

    for user in test_users:
        print(f"\nInterviewing {user.name} ({user.role})...")

        session = InterviewSession(user=user)
        result = await program(session)

        print(f"  ✓ Questions: {result.questions_asked}")
        print(f"  ✓ Requirements: {len(result.session.requirements)}")
        print(f"  ✓ Confidence: {result.confidence_score:.0%}")

    print("\n✅ Batch interviews complete")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "batch":
        asyncio.run(batch_interviews())
    else:
        asyncio.run(interactive_interview())
