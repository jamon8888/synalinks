"""
Complete Interview → Plugin Generation → Marketplace Setup

End-to-end automation: Interview user → Generate PRD → Create plugins → Package for distribution
"""

import asyncio
import sys
from pathlib import Path

import synalinks
from synalinks.src.modules.interview_agent import (
    InterviewAgent,
    PRDGenerator,
    PluginMarketplace,
    User,
    InterviewSession,
)
from synalinks.src.modules.interview_agent.plugin_packaging import (
    PluginGenerator,
    MarketplaceBuilder,
)


async def complete_interview_to_plugins_flow(
    user_info: dict = None,
    output_dir: str = "./interview_output"
):
    """
    Complete flow from interview to installable plugins.

    Steps:
    1. Conduct interview
    2. Generate PRD
    3. Create plugins from requirements
    4. Package into marketplace
    5. Generate installation instructions

    Args:
        user_info: Optional pre-filled user information
        output_dir: Where to save generated artifacts
    """
    print("=" * 70)
    print("INTERVIEW → PLUGIN GENERATION → MARKETPLACE SETUP")
    print("=" * 70)

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # ========================================================================
    # STEP 1: CONDUCT INTERVIEW
    # ========================================================================

    print("\n📋 STEP 1: Interview Phase")
    print("-" * 70)

    # Get user info
    if not user_info:
        user_info = await get_user_info_interactive()

    user = User(
        label="User",
        name=user_info["name"],
        role=user_info["role"],
        experience_level=user_info["experience"],
        industry=user_info.get("industry"),
    )

    # Setup interview agent
    lm = synalinks.LanguageModel(model="anthropic/claude-sonnet-4")
    kb = synalinks.KnowledgeBase(uri="neo4j://localhost:7687")

    interview_agent = InterviewAgent(
        language_model=lm,
        knowledge_base=kb,
        max_questions=15,
    )

    await interview_agent.build(input_shape=(InterviewSession,))

    # Run interview
    session = InterviewSession(user=user)

    # For demo, use mock data (replace with real interview)
    session = await mock_interview_session(user)

    print(f"✓ Interview complete")
    print(f"  Requirements discovered: {len(session.requirements)}")
    print(f"  Use cases: {len(session.use_cases)}")

    # ========================================================================
    # STEP 2: GENERATE PRD
    # ========================================================================

    print("\n📄 STEP 2: PRD Generation")
    print("-" * 70)

    prd_generator = PRDGenerator(language_model=lm, knowledge_base=kb)
    await prd_generator.build(input_shape=(InterviewSession,))

    prd = await prd_generator(session)

    # Save PRD
    prd_filename = output_path / f"prd_{user.name.replace(' ', '_').lower()}.md"
    prd_markdown = prd_generator.format_prd_markdown(prd)

    with open(prd_filename, "w") as f:
        f.write(prd_markdown)

    print(f"✓ PRD generated")
    print(f"  Total requirements: {len(prd.requirements.all_requirements)}")
    print(f"  Categories: {list(prd.requirements.by_category.keys())}")
    print(f"  Saved to: {prd_filename}")

    # ========================================================================
    # STEP 3: GENERATE PLUGINS
    # ========================================================================

    print("\n🔌 STEP 3: Plugin Generation")
    print("-" * 70)

    plugins_dir = output_path / "plugins"
    plugin_generator = PluginGenerator(output_dir=str(plugins_dir))

    # Generate plugins from high-priority requirements
    generated_plugins = []

    for req in prd.requirements.all_requirements:
        if req.priority >= 4:  # Only high priority
            try:
                plugin_path = plugin_generator.generate_plugin_from_requirement(
                    requirement=req,
                    user=user,
                )
                generated_plugins.append(plugin_path.name)
                print(f"  ✓ Generated: {plugin_path.name}")

            except Exception as e:
                print(f"  ⚠ Error with {req.name}: {e}")

    print(f"\n✓ Generated {len(generated_plugins)} plugins")

    # ========================================================================
    # STEP 4: CREATE MARKETPLACE
    # ========================================================================

    print("\n🏪 STEP 4: Marketplace Creation")
    print("-" * 70)

    marketplace_name = f"{user.name.replace(' ', '-').lower()}-plugins"
    marketplace_builder = MarketplaceBuilder(marketplace_dir=str(output_path / "marketplace"))

    marketplace_path = marketplace_builder.create_marketplace_repo(
        marketplace_name=marketplace_name,
        plugins_dir=plugins_dir,
        prd=prd,
    )

    print(f"✓ Marketplace created at: {marketplace_path}")

    # ========================================================================
    # STEP 5: GENERATE INSTALLATION GUIDE
    # ========================================================================

    print("\n📦 STEP 5: Installation Instructions")
    print("-" * 70)

    install_instructions = generate_installation_instructions(
        user=user,
        marketplace_name=marketplace_name,
        plugins=generated_plugins,
        marketplace_path=marketplace_path,
    )

    install_file = output_path / "INSTALLATION.md"
    with open(install_file, "w") as f:
        f.write(install_instructions)

    print(f"✓ Installation guide: {install_file}")

    # ========================================================================
    # STEP 6: CREATE SETUP SCRIPTS
    # ========================================================================

    print("\n⚡ STEP 6: Setup Automation")
    print("-" * 70)

    # Create git repository setup script
    git_setup_script = create_git_setup_script(
        marketplace_path=marketplace_path,
        marketplace_name=marketplace_name,
    )

    git_script_file = output_path / "setup_github_repo.sh"
    with open(git_script_file, "w") as f:
        f.write(git_setup_script)

    git_script_file.chmod(0o755)  # Make executable

    print(f"✓ GitHub setup script: {git_script_file}")

    # Create quick install script for users
    quick_install = create_quick_install_script(
        marketplace_name=marketplace_name,
        plugins=generated_plugins,
    )

    quick_install_file = marketplace_path / "quick-install.sh"
    with open(quick_install_file, "w") as f:
        f.write(quick_install)

    quick_install_file.chmod(0o755)

    print(f"✓ Quick install script: {quick_install_file}")

    # ========================================================================
    # SUMMARY
    # ========================================================================

    print("\n" + "=" * 70)
    print("✅ COMPLETE PIPELINE FINISHED")
    print("=" * 70)

    print(f"\nGenerated Artifacts:")
    print(f"  📄 PRD: {prd_filename}")
    print(f"  🔌 Plugins: {plugins_dir} ({len(generated_plugins)} plugins)")
    print(f"  🏪 Marketplace: {marketplace_path}")
    print(f"  📦 Installation Guide: {install_file}")
    print(f"  ⚡ Setup Scripts: {git_script_file}")

    print(f"\n🚀 Next Steps:")
    print(f"\n1. Push to GitHub:")
    print(f"   cd {marketplace_path}")
    print(f"   bash ../setup_github_repo.sh")

    print(f"\n2. Users can install via:")
    print(f"   /plugin marketplace add [your-github-username]/{marketplace_name}")

    for plugin in generated_plugins[:3]:
        print(f"   /plugin install {plugin}@{marketplace_name}")

    print(f"\n3. Or use quick install:")
    print(f"   bash {quick_install_file}")

    return {
        "prd": prd,
        "plugins": generated_plugins,
        "marketplace_path": marketplace_path,
        "output_dir": output_path,
    }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


async def get_user_info_interactive():
    """Get user information interactively."""
    print("\nUser Information")
    print("-" * 70)

    name = input("Name: ").strip() or "User"
    role = input("Role (e.g., Software Engineer): ").strip() or "Developer"
    experience = input("Experience (beginner/intermediate/advanced): ").strip() or "intermediate"
    industry = input("Industry (optional): ").strip() or None

    return {
        "name": name,
        "role": role,
        "experience": experience,
        "industry": industry,
    }


async def mock_interview_session(user: User) -> InterviewSession:
    """
    Create a mock interview session with sample data.

    In production, this would be replaced with actual interview results.
    """
    from synalinks.src.modules.interview_agent import Requirement, UseCase, Technology, Goal
    from datetime import datetime

    session = InterviewSession(
        user=user,
        requirements=[
            Requirement(
                label="Requirement",
                name="Automated Code Review",
                description="Automated PR reviews with best practices checking",
                priority=5,
                category="coding",
                confidence_score=0.9,
                discovered_at=datetime.now().isoformat(),
            ),
            Requirement(
                label="Requirement",
                name="Test Generation",
                description="Automatically generate unit tests from code",
                priority=4,
                category="testing",
                confidence_score=0.85,
                discovered_at=datetime.now().isoformat(),
            ),
            Requirement(
                label="Requirement",
                name="Documentation Generation",
                description="Auto-generate API docs and code comments",
                priority=4,
                category="documentation",
                confidence_score=0.8,
                discovered_at=datetime.now().isoformat(),
            ),
        ],
        use_cases=[
            UseCase(
                label="UseCase",
                name="Daily PR Review",
                description="Review 5-10 PRs per day for code quality",
                frequency="daily",
                pain_points=["Manual process takes 2 hours", "Inconsistent review standards"],
            ),
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
                success_criteria=["30% fewer bugs", "Faster PR reviews"],
            ),
        ],
        questions_asked=10,
    )

    return session


def generate_installation_instructions(
    user: User,
    marketplace_name: str,
    plugins: list,
    marketplace_path: Path,
) -> str:
    """Generate detailed installation instructions."""

    return f"""# Installation Instructions

Welcome {user.name}! Your personalized Claude Code plugins are ready to install.

## Prerequisites

- Claude Code installed and configured
- GitHub account (to host the marketplace)

## Quick Install (After GitHub Setup)

```bash
# 1. Add the marketplace
/plugin marketplace add [your-github-username]/{marketplace_name}

# 2. Install all recommended plugins
{chr(10).join(f'/plugin install {p}@{marketplace_name}' for p in plugins)}
```

## Step-by-Step Setup

### Step 1: Push to GitHub

```bash
cd {marketplace_path}

# Initialize git (if not already done)
git init
git add .
git commit -m "Add interview-generated plugins"

# Create GitHub repo (using gh CLI)
gh repo create {marketplace_name} --public --source=. --push

# Or manually:
# 1. Create repo on github.com
# 2. git remote add origin https://github.com/[username]/{marketplace_name}.git
# 3. git push -u origin main
```

### Step 2: Add Marketplace in Claude Code

```bash
/plugin marketplace add [your-github-username]/{marketplace_name}
```

### Step 3: Browse and Install Plugins

```bash
# See all available plugins
/plugin list

# Install specific plugins
{chr(10).join(f'/plugin install {p}@{marketplace_name}' for p in plugins[:3])}
```

### Step 4: Verify Installation

```bash
/plugin list --installed
```

## Alternative: Project-Level Configuration

Add to your project's `.claude/settings.json`:

```json
{{
  "plugins": {{
{chr(10).join(f'    "{p}": {{"marketplace": "[username]/{marketplace_name}", "version": "1.0.0"}},' for p in plugins)}
  }}
}}
```

## Plugins Included

{chr(10).join(f'### {i+1}. {p}{chr(10)}Purpose: [See {p}/README.md for details]{chr(10)}' for i, p in enumerate(plugins))}

## Need Help?

- Check individual plugin READMEs in `{marketplace_path}/[plugin-name]/README.md`
- See [INSTALL.md]({marketplace_path}/INSTALL.md) in the marketplace
- Review Claude Code docs: https://code.claude.com/docs

## Updates

To update plugins in the future:
1. Re-run the interview to discover new needs
2. Regenerate plugins
3. Push updates to GitHub
4. Users will be notified of available updates

---

Generated by Interview Agent
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""


def create_git_setup_script(marketplace_path: Path, marketplace_name: str) -> str:
    """Create script to setup GitHub repository."""

    return f"""#!/bin/bash
# GitHub Repository Setup for {marketplace_name}
# Run this script from the marketplace directory

set -e

echo "Setting up GitHub repository for {marketplace_name}..."

# Check if we're in the right directory
if [ ! -f ".claude-plugin/marketplace.json" ]; then
    echo "Error: marketplace.json not found. Run from marketplace directory."
    exit 1
fi

# Initialize git if needed
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
    git add .
    git commit -m "Initial commit: Interview-generated plugin marketplace"
fi

# Check if gh CLI is installed
if command -v gh &> /dev/null; then
    echo "Using GitHub CLI to create repository..."

    # Create public repository and push
    gh repo create {marketplace_name} --public --source=. --push --description "Personalized Claude Code plugins from interview"

    echo "✓ Repository created and pushed to GitHub"
    echo ""
    echo "Next steps:"
    echo "1. Add marketplace in Claude Code:"
    echo "   /plugin marketplace add $(git config --get remote.origin.url | sed 's|https://github.com/||' | sed 's|.git||')/{marketplace_name}"
    echo ""
    echo "2. Install plugins:"
    echo "   /plugin install [plugin-name]@{marketplace_name}"

else
    echo "GitHub CLI (gh) not found."
    echo ""
    echo "Manual setup required:"
    echo "1. Go to https://github.com/new"
    echo "2. Create a repository named: {marketplace_name}"
    echo "3. Make it public"
    echo "4. Run these commands:"
    echo ""
    echo "   git remote add origin https://github.com/[your-username]/{marketplace_name}.git"
    echo "   git branch -M main"
    echo "   git push -u origin main"
    echo ""
    echo "Then add marketplace in Claude Code:"
    echo "   /plugin marketplace add [your-username]/{marketplace_name}"
fi
"""


def create_quick_install_script(marketplace_name: str, plugins: list) -> str:
    """Create quick install script for end users."""

    return f"""#!/bin/bash
# Quick Install Script for {marketplace_name}
#
# This script helps users quickly install all recommended plugins
# Usage: bash quick-install.sh [github-username]

USERNAME=${{1:-"your-username"}}

echo "Installing Claude Code plugins from {marketplace_name}"
echo "GitHub user: $USERNAME"
echo ""

# Check if Claude Code is available
if ! command -v claude &> /dev/null; then
    echo "Warning: Claude Code CLI not found in PATH"
    echo "You can still use /plugin commands in Claude Code UI"
    echo ""
fi

echo "Add the following commands to Claude Code:"
echo ""
echo "/plugin marketplace add $USERNAME/{marketplace_name}"
echo ""

# Generate install commands
{chr(10).join(f'echo "/plugin install {p}@{marketplace_name}"' for p in plugins)}

echo ""
echo "Or copy this complete installation block:"
echo ""
echo "/plugin marketplace add $USERNAME/{marketplace_name}"
{chr(10).join(f'echo "/plugin install {p}@{marketplace_name}"' for p in plugins)}

echo ""
echo "Installation complete! Check /plugin list --installed"
"""


# ============================================================================
# MAIN
# ============================================================================


async def main():
    """Main entry point."""

    import argparse

    parser = argparse.ArgumentParser(description="Generate Claude Code plugins from interview")
    parser.add_argument("--name", help="User name")
    parser.add_argument("--role", help="User role")
    parser.add_argument("--experience", help="Experience level", default="intermediate")
    parser.add_argument("--industry", help="Industry")
    parser.add_argument("--output", help="Output directory", default="./interview_output")

    args = parser.parse_args()

    # Prepare user info
    user_info = None
    if args.name and args.role:
        user_info = {
            "name": args.name,
            "role": args.role,
            "experience": args.experience,
            "industry": args.industry,
        }

    # Run complete flow
    try:
        result = await complete_interview_to_plugins_flow(
            user_info=user_info,
            output_dir=args.output,
        )

        print("\n✅ Success! All artifacts generated.")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
"""
