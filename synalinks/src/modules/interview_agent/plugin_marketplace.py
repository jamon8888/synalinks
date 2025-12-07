"""
Plugin Marketplace

Manages plugin discovery, recommendation, and bundle creation based on
user requirements and PRDs.
"""

from typing import List, Dict, Optional, Set, Tuple
from pathlib import Path
import json
from collections import defaultdict

from .data_models import (
    Plugin,
    Requirement,
    PRD,
    PluginRecommendation,
    PluginBundle,
    OrganizedRequirements,
    PluginSatisfiesRequirement,
)


class PluginMarketplace:
    """
    Manages Claude Code plugin recommendations and bundles.

    Features:
    - Index plugins from repository/directory
    - Semantic matching of requirements to plugins
    - Graph-based collaborative filtering (users with similar needs)
    - Bundle optimization (maximize coverage, minimize redundancy)
    - Installation script generation

    Args:
        knowledge_base: Graph database with plugin and requirement data
        language_model: LM for semantic matching and reasoning
    """

    def __init__(self, knowledge_base, language_model):
        self.knowledge_base = knowledge_base
        self.language_model = language_model
        self.indexed_plugins: Dict[str, Plugin] = {}

    async def index_plugins(self, plugin_directory: str):
        """
        Index all available Claude Code plugins into knowledge graph.

        Scans directory for plugin metadata and stores in knowledge base.

        Args:
            plugin_directory: Path to directory containing plugins
        """
        plugin_path = Path(plugin_directory)

        if not plugin_path.exists():
            raise ValueError(f"Plugin directory does not exist: {plugin_directory}")

        # Scan for plugin metadata files
        plugins = await self._discover_plugins(plugin_path)

        # Add to knowledge base
        if plugins:
            await self.knowledge_base.add_entities(plugins)

        # Cache locally
        for plugin in plugins:
            self.indexed_plugins[plugin.name] = plugin

        return len(plugins)

    async def _discover_plugins(self, plugin_path: Path) -> List[Plugin]:
        """
        Discover plugins from directory structure.

        Expected structure:
        plugins/
          plugin-name/
            plugin.json  (metadata: name, description, capabilities, categories)
            README.md
            ...
        """
        plugins = []

        # Look for plugin.json files
        for metadata_file in plugin_path.rglob("plugin.json"):
            try:
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)

                plugin = Plugin(
                    label="Plugin",
                    name=metadata.get("name", metadata_file.parent.name),
                    description=metadata.get("description", ""),
                    capabilities=metadata.get("capabilities", []),
                    categories=metadata.get("categories", []),
                    install_count=metadata.get("install_count", 0),
                    rating=metadata.get("rating", 0.0),
                    repository_url=metadata.get("repository_url"),
                    author=metadata.get("author"),
                    dependencies=metadata.get("dependencies", []),
                )

                plugins.append(plugin)

            except Exception as e:
                print(f"Error reading plugin metadata from {metadata_file}: {e}")
                continue

        return plugins

    async def recommend_plugins(
        self, prd: PRD, top_k: int = 10
    ) -> List[PluginRecommendation]:
        """
        Recommend plugins based on PRD requirements.

        Uses hybrid approach:
        1. Semantic search (requirement descriptions → plugin capabilities)
        2. Graph-based (find plugins used by similar users)
        3. Category matching
        4. Collaborative filtering

        Args:
            prd: Product Requirements Document
            top_k: Number of recommendations to return

        Returns:
            Ranked list of plugin recommendations
        """
        all_recommendations = []

        # Process each requirement
        for requirement in prd.requirements.all_requirements:
            # Semantic matching
            semantic_matches = await self._semantic_match_plugins(requirement)

            # Graph-based matching (similar users)
            graph_matches = await self._graph_based_match(requirement, prd.user)

            # Category matching
            category_matches = await self._category_match(requirement)

            # Combine and score
            combined = self._combine_matches(
                requirement=requirement,
                semantic=semantic_matches,
                graph=graph_matches,
                category=category_matches,
            )

            all_recommendations.extend(combined)

        # Deduplicate and rank
        final_recommendations = self._rank_and_deduplicate(
            all_recommendations, prd, top_k
        )

        return final_recommendations

    async def _semantic_match_plugins(
        self, requirement: Requirement, top_k: int = 5
    ) -> List[Tuple[Plugin, float]]:
        """
        Find plugins semantically similar to requirement.

        Uses embedding similarity between requirement description and
        plugin capabilities.
        """
        try:
            # Query knowledge base for similar plugins
            # Would use vector similarity search
            query_text = f"{requirement.name}: {requirement.description}"

            # Placeholder: return empty for now
            # In production:
            # similar = await self.knowledge_base.query_similar(
            #     entity_type=Plugin,
            #     query_text=query_text,
            #     top_k=top_k
            # )

            return []

        except Exception:
            return []

    async def _graph_based_match(
        self, requirement: Requirement, user, top_k: int = 5
    ) -> List[Tuple[Plugin, float]]:
        """
        Find plugins used by users with similar requirements (collaborative filtering).

        Cypher query concept:
        MATCH (u1:User)-[:HasRequirement]->(r1:Requirement {category: $category})
        MATCH (u2:User)-[:HasRequirement]->(r2:Requirement {category: $category})
        WHERE u1 <> u2
        MATCH (p:Plugin)-[:PluginSatisfiesRequirement]->(r2)
        WITH p, count(*) as usage_count, avg(p.rating) as avg_rating
        ORDER BY usage_count DESC, avg_rating DESC
        LIMIT $top_k
        RETURN p, usage_count
        """
        try:
            # Placeholder: return empty
            # In production would query knowledge graph
            return []

        except Exception:
            return []

    async def _category_match(
        self, requirement: Requirement, top_k: int = 5
    ) -> List[Tuple[Plugin, float]]:
        """Find plugins that match the requirement category."""
        matches = []

        for plugin in self.indexed_plugins.values():
            if requirement.category in plugin.categories:
                # Simple category match score
                score = 0.5 + (plugin.rating / 10.0)  # 0.5-1.0 range
                matches.append((plugin, score))

        # Sort by score
        matches.sort(key=lambda x: x[1], reverse=True)

        return matches[:top_k]

    def _combine_matches(
        self,
        requirement: Requirement,
        semantic: List[Tuple[Plugin, float]],
        graph: List[Tuple[Plugin, float]],
        category: List[Tuple[Plugin, float]],
    ) -> List[PluginRecommendation]:
        """
        Combine different match sources with weighted scoring.

        Weights:
        - Semantic: 0.4 (most important - actual capability match)
        - Graph: 0.35 (social proof - what similar users use)
        - Category: 0.25 (basic compatibility)
        """
        plugin_scores = defaultdict(lambda: {"scores": [], "sources": set()})

        # Collect scores from each source
        for plugin, score in semantic:
            plugin_scores[plugin.name]["scores"].append(("semantic", score * 0.4))
            plugin_scores[plugin.name]["plugin"] = plugin
            plugin_scores[plugin.name]["sources"].add("semantic")

        for plugin, score in graph:
            plugin_scores[plugin.name]["scores"].append(("graph", score * 0.35))
            plugin_scores[plugin.name]["plugin"] = plugin
            plugin_scores[plugin.name]["sources"].add("graph")

        for plugin, score in category:
            plugin_scores[plugin.name]["scores"].append(("category", score * 0.25))
            plugin_scores[plugin.name]["plugin"] = plugin
            plugin_scores[plugin.name]["sources"].add("category")

        # Build recommendations
        recommendations = []
        for plugin_name, data in plugin_scores.items():
            if "plugin" not in data:
                continue

            # Total score
            total_score = sum(score for _, score in data["scores"])

            # Boost if matched by multiple sources
            if len(data["sources"]) > 1:
                total_score *= 1.1

            # Generate reasoning
            reasoning = self._generate_match_reasoning(
                requirement=requirement,
                plugin=data["plugin"],
                sources=data["sources"],
                score=total_score,
            )

            rec = PluginRecommendation(
                plugin=data["plugin"],
                relevance_score=min(1.0, total_score),
                matched_requirements=[requirement],
                matched_use_cases=[],
                reasoning=reasoning,
                installation_priority=self._calculate_priority(requirement, total_score),
            )

            recommendations.append(rec)

        return recommendations

    def _generate_match_reasoning(
        self,
        requirement: Requirement,
        plugin: Plugin,
        sources: Set[str],
        score: float,
    ) -> str:
        """Generate human-readable reasoning for why plugin matches requirement."""
        reasons = []

        # Category match
        if requirement.category in plugin.categories:
            reasons.append(
                f"Addresses {requirement.category} needs"
            )

        # Capability overlap
        req_words = set(requirement.description.lower().split())
        cap_words = set(" ".join(plugin.capabilities).lower().split())
        overlap = req_words & cap_words
        if overlap:
            key_matches = list(overlap)[:3]
            reasons.append(f"Provides capabilities: {', '.join(key_matches)}")

        # Rating
        if plugin.rating >= 4.0:
            reasons.append(f"Highly rated ({plugin.rating:.1f}/5)")

        # Sources
        if "semantic" in sources:
            reasons.append("Semantically similar to requirement")
        if "graph" in sources:
            reasons.append("Used by users with similar needs")

        # Default
        if not reasons:
            reasons.append("General capability match")

        return f"{plugin.description}. " + ". ".join(reasons) + "."

    def _calculate_priority(self, requirement: Requirement, match_score: float) -> int:
        """
        Calculate installation priority (1-10).

        Based on requirement priority and match quality.
        """
        # Requirement priority (1-5) maps to base priority
        base_priority = requirement.priority * 2  # Now 2-10

        # Adjust by match score
        adjusted = base_priority * match_score

        return min(10, max(1, int(adjusted)))

    def _rank_and_deduplicate(
        self,
        recommendations: List[PluginRecommendation],
        prd: PRD,
        top_k: int,
    ) -> List[PluginRecommendation]:
        """
        Deduplicate plugins and rank by overall value.

        Combines:
        - Relevance score
        - Number of requirements satisfied
        - Installation priority
        - Plugin rating
        """
        # Group by plugin
        plugin_to_recs = defaultdict(list)
        for rec in recommendations:
            plugin_to_recs[rec.plugin.name].append(rec)

        # Merge recommendations for same plugin
        merged = []
        for plugin_name, recs in plugin_to_recs.items():
            # Take best recommendation
            best_rec = max(recs, key=lambda r: r.relevance_score)

            # Aggregate requirements
            all_reqs = []
            for rec in recs:
                all_reqs.extend(rec.matched_requirements)

            # Deduplicate requirements
            unique_reqs = list({req.name: req for req in all_reqs}.values())

            # Update recommendation
            best_rec.matched_requirements = unique_reqs

            # Boost score for multi-requirement plugins
            coverage_boost = min(0.2, len(unique_reqs) * 0.05)
            best_rec.relevance_score = min(1.0, best_rec.relevance_score + coverage_boost)

            merged.append(best_rec)

        # Sort by composite score
        def composite_score(rec: PluginRecommendation) -> float:
            return (
                rec.relevance_score * 0.5  # Match quality
                + (len(rec.matched_requirements) / 10) * 0.3  # Coverage
                + (rec.plugin.rating / 5) * 0.2  # Plugin quality
            )

        merged.sort(key=composite_score, reverse=True)

        return merged[:top_k]

    async def create_personalized_bundle(
        self, prd: PRD, max_plugins: int = 10
    ) -> PluginBundle:
        """
        Create an optimized bundle of plugins for a user.

        Optimization goals:
        1. Maximize requirement coverage
        2. Minimize redundancy
        3. Respect dependencies
        4. Balance installation complexity

        Args:
            prd: Product Requirements Document
            max_plugins: Maximum plugins in bundle

        Returns:
            Optimized plugin bundle with install script
        """
        # Get recommendations
        recommendations = await self.recommend_plugins(prd, top_k=max_plugins * 2)

        # Optimize bundle (greedy set cover approach)
        bundle_plugins = self._optimize_bundle(
            recommendations=recommendations,
            all_requirements=prd.requirements.all_requirements,
            max_plugins=max_plugins,
        )

        # Generate installation script
        install_script = self._generate_install_script(bundle_plugins)

        # Generate configuration guide
        config_guide = self._generate_config_guide(
            bundle_plugins=bundle_plugins, prd=prd
        )

        # Calculate coverage
        coverage = self._calculate_coverage(
            bundle_plugins=bundle_plugins,
            all_requirements=prd.requirements.all_requirements,
        )

        # Estimate setup time
        setup_time = self._estimate_setup_time(bundle_plugins)

        return PluginBundle(
            plugins=[rec.plugin for rec in bundle_plugins],
            install_script=install_script,
            configuration_guide=config_guide,
            estimated_setup_time=setup_time,
            coverage_score=coverage,
        )

    def _optimize_bundle(
        self,
        recommendations: List[PluginRecommendation],
        all_requirements: List[Requirement],
        max_plugins: int,
    ) -> List[PluginRecommendation]:
        """
        Optimize plugin selection using greedy set cover.

        Iteratively select plugins that cover the most uncovered requirements.
        """
        selected = []
        covered_requirements = set()
        all_req_names = {req.name for req in all_requirements}

        # Sort by coverage potential
        sorted_recs = sorted(
            recommendations,
            key=lambda r: (
                len(r.matched_requirements),
                r.relevance_score,
                r.plugin.rating,
            ),
            reverse=True,
        )

        for rec in sorted_recs:
            if len(selected) >= max_plugins:
                break

            # Check how many new requirements this covers
            new_coverage = set(
                req.name for req in rec.matched_requirements
            ) - covered_requirements

            # Add if it provides new coverage or is highly rated
            if new_coverage or (rec.relevance_score > 0.8 and len(selected) < 5):
                selected.append(rec)
                covered_requirements.update(req.name for req in rec.matched_requirements)

            # Stop if we've covered everything
            if covered_requirements >= all_req_names:
                break

        return selected

    def _generate_install_script(self, bundle: List[PluginRecommendation]) -> str:
        """Generate shell script to install all plugins in bundle."""
        lines = [
            "#!/bin/bash",
            "# Claude Code Plugin Installation Script",
            "# Generated by Interview Agent",
            "",
            "set -e  # Exit on error",
            "",
            "echo 'Installing personalized Claude Code plugin bundle...'",
            "",
        ]

        for i, rec in enumerate(bundle, 1):
            plugin = rec.plugin
            lines.append(f"# {i}. {plugin.name}")
            lines.append(f"# {plugin.description}")
            lines.append(f"echo 'Installing {plugin.name}...'")

            if plugin.repository_url:
                # Clone from repository
                lines.append(
                    f"git clone {plugin.repository_url} ~/.claude/plugins/{plugin.name}"
                )
            else:
                # Placeholder for other installation methods
                lines.append(f"# TODO: Install {plugin.name}")

            # Install dependencies if any
            if plugin.dependencies:
                lines.append(f"# Dependencies: {', '.join(plugin.dependencies)}")
                for dep in plugin.dependencies:
                    lines.append(f"npm install -g {dep} || pip install {dep} || true")

            lines.append("")

        lines.append("echo 'All plugins installed successfully!'")
        lines.append("echo 'Run the configuration guide for setup instructions.'")

        return "\n".join(lines)

    def _generate_config_guide(
        self, bundle_plugins: List[PluginRecommendation], prd: PRD
    ) -> str:
        """Generate configuration guide for installed plugins."""
        lines = [
            f"# Configuration Guide - {prd.user.name}",
            "",
            f"Personalized setup for {len(bundle_plugins)} plugins.",
            "",
            "## Installed Plugins",
            "",
        ]

        for i, rec in enumerate(bundle_plugins, 1):
            plugin = rec.plugin
            lines.append(f"### {i}. {plugin.name}")
            lines.append(f"{plugin.description}")
            lines.append("")
            lines.append(f"**Satisfies:** {', '.join(req.name for req in rec.matched_requirements)}")
            lines.append(f"**Priority:** {rec.installation_priority}/10")
            lines.append("")
            lines.append("**Configuration:**")
            lines.append(f"- Edit `~/.claude/plugins/{plugin.name}/config.json`")
            lines.append("- Set your preferences based on use cases")
            lines.append("")

        lines.append("## Next Steps")
        lines.append("")
        lines.append("1. Review and test each plugin")
        lines.append("2. Customize configurations based on your workflow")
        lines.append("3. Provide feedback to improve recommendations")

        return "\n".join(lines)

    def _calculate_coverage(
        self,
        bundle_plugins: List[PluginRecommendation],
        all_requirements: List[Requirement],
    ) -> float:
        """Calculate what percentage of requirements are covered by bundle."""
        if not all_requirements:
            return 0.0

        covered = set()
        for rec in bundle_plugins:
            covered.update(req.name for req in rec.matched_requirements)

        total = len(all_requirements)
        coverage = len(covered) / total

        return round(coverage, 2)

    def _estimate_setup_time(self, bundle: List[PluginRecommendation]) -> str:
        """Estimate total setup time for bundle."""
        # Rough estimate: 5 min per plugin + 2 min per dependency
        base_time = len(bundle) * 5
        dep_time = sum(
            len(rec.plugin.dependencies) * 2 for rec in bundle
        )

        total_minutes = base_time + dep_time

        if total_minutes < 60:
            return f"{total_minutes} minutes"
        else:
            hours = total_minutes // 60
            minutes = total_minutes % 60
            return f"{hours}h {minutes}m"
