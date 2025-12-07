

# Plugin Generation & Packaging Guide

Complete guide for generating, packaging, and distributing Claude Code plugins from interview discoveries.

## Overview

After conducting an interview and generating a PRD, you can automatically create installable Claude Code plugins that address the user's discovered needs.

```
Interview → PRD → Plugin Generation → Marketplace → Easy Installation
```

## Quick Start

### One Command Setup

```bash
cd /home/user/synalinks

# Run complete flow (interactive)
python examples/generate_plugins_from_interview.py

# Or with pre-filled info
python examples/generate_plugins_from_interview.py \
  --name "Alice" \
  --role "Software Engineer" \
  --experience "advanced" \
  --industry "FinTech" \
  --output "./my_plugins"
```

This will:
1. ✅ Conduct interview (or use mock data)
2. ✅ Generate PRD
3. ✅ Create plugins for each high-priority requirement
4. ✅ Package into a marketplace
5. ✅ Generate installation scripts
6. ✅ Create GitHub setup automation

## What Gets Generated

### Directory Structure

```
interview_output/
├── prd_alice.md                           # Product Requirements Document
├── INSTALLATION.md                        # User installation guide
├── setup_github_repo.sh                   # GitHub automation script
│
├── plugins/                               # Individual plugins
│   ├── automated-code-review/
│   │   ├── .claude-plugin/
│   │   │   └── plugin.json                # Plugin manifest
│   │   ├── commands/                      # Slash commands
│   │   │   └── code_review.md
│   │   ├── README.md
│   │   └── CLAUDE.md
│   │
│   ├── test-generation/
│   │   ├── .claude-plugin/
│   │   │   └── plugin.json
│   │   ├── skills/
│   │   │   └── test-generation/
│   │   │       └── SKILL.md               # Reusable skill
│   │   ├── README.md
│   │   └── CLAUDE.md
│   │
│   └── documentation-generation/
│       ├── .claude-plugin/
│       │   └── plugin.json
│       ├── .mcp.json                      # MCP server config
│       ├── server.js                      # MCP server implementation
│       ├── README.md
│       └── CLAUDE.md
│
└── marketplace/
    └── alice-plugins/                     # GitHub-ready marketplace
        ├── .claude-plugin/
        │   └── marketplace.json           # Marketplace manifest
        ├── automated-code-review/         # Copy of plugin
        ├── test-generation/
        ├── documentation-generation/
        ├── README.md                      # Marketplace README
        ├── INSTALL.md                     # Installation guide
        ├── CLAUDE.md                      # Context for Claude
        └── quick-install.sh               # User install script
```

## Plugin Types Generated

Based on requirement categories, different plugin types are created:

| Requirement Category | Plugin Type | Components |
|---------------------|-------------|------------|
| **automation** | Command Plugin | `/commands/*.md` |
| **coding** | Skill Plugin | `/skills/*/SKILL.md` |
| **testing** | Command Plugin | `/commands/*.md` |
| **documentation** | Command Plugin | `/commands/*.md` |
| **data_analysis** | MCP Plugin | `.mcp.json`, `server.js` |
| **integration** | MCP Plugin | `.mcp.json`, `server.js` |

## Plugin Structure Examples

### 1. Command Plugin (Slash Commands)

**Generated for:** Automation, Testing, Documentation needs

**Structure:**
```
automated-code-review/
├── .claude-plugin/
│   └── plugin.json
├── commands/
│   └── code_review.md          # The slash command
├── README.md
└── CLAUDE.md
```

**Command File** (`commands/code_review.md`):
```markdown
---
description: Automated code review with best practices
---

# code_review

Review the current code or PR for:
- Code quality issues
- Best practice violations
- Security concerns
- Performance opportunities

## Usage

```
/code_review
/code_review --file src/main.py
/code_review --pr 123
```

## Examples

- `/code_review` - Review current file
- `/code_review --pr 123` - Review PR #123
```

**User Invokes:** `/code_review` in Claude Code

### 2. Skill Plugin (Reusable Capabilities)

**Generated for:** Coding, Development needs

**Structure:**
```
test-generation/
├── .claude-plugin/
│   └── plugin.json
├── skills/
│   └── test-generation/
│       └── SKILL.md            # The skill
├── README.md
└── CLAUDE.md
```

**Skill File** (`skills/test-generation/SKILL.md`):
```markdown
---
name: test-generation
description: Generate comprehensive unit tests
---

# Test Generation Skill

## Capability
Automatically generate unit tests from existing code, including edge cases and mocks.

## When to Use
- When creating new code that needs tests
- When adding tests to legacy code
- When expanding test coverage

## Examples
- "Generate tests for this function"
- "Create unit tests with edge cases"
- "Add integration tests for this API"
```

**Usage:** Agent automatically uses this skill when relevant

### 3. MCP Server Plugin (External Tools)

**Generated for:** Data Analysis, Integrations

**Structure:**
```
data-pipeline/
├── .claude-plugin/
│   └── plugin.json
├── .mcp.json                   # MCP server config
├── server.js                   # Server implementation
├── package.json                # Node dependencies
├── README.md
└── CLAUDE.md
```

**.mcp.json:**
```json
{
  "servers": {
    "data-pipeline-server": {
      "command": "node",
      "args": ["${CLAUDE_PLUGIN_ROOT}/server.js"],
      "transport": "stdio",
      "env": {
        "API_KEY": "${DATA_API_KEY}"
      }
    }
  }
}
```

**server.js:** (Starter template)
```javascript
const { Server } = require('@modelcontextprotocol/sdk/server');
const { StdioServerTransport } = require('@modelcontextprotocol/sdk/server/stdio');

const server = new Server({
  name: 'data-pipeline-server',
  version: '1.0.0'
});

// Add tools, prompts, resources here

const transport = new StdioServerTransport();
server.connect(transport);
```

**Usage:** MCP server auto-starts, provides tools to Claude Code

## Marketplace Structure

### marketplace.json

**Location:** `.claude-plugin/marketplace.json`

```json
{
  "name": "Interview Recommended Plugins for Alice",
  "description": "Personalized plugins from AI interview",
  "author": {
    "name": "Alice"
  },
  "plugins": [
    {
      "name": "automated-code-review",
      "displayName": "Automated Code Review",
      "description": "Automated PR reviews with best practices",
      "version": "1.0.0",
      "keywords": ["coding", "review"],
      "source": "./automated-code-review"
    },
    {
      "name": "test-generation",
      "displayName": "Test Generation",
      "description": "Generate unit tests automatically",
      "version": "1.0.0",
      "keywords": ["testing"],
      "source": "./test-generation"
    }
  ]
}
```

### Marketplace README

Automatically generated with:
- User profile information
- List of included plugins
- Installation instructions
- Requirements addressed
- Generated date

## Installation Flow

### For Plugin Developers (You)

```bash
# 1. Generate plugins from interview
python examples/generate_plugins_from_interview.py

# 2. Navigate to marketplace
cd interview_output/marketplace/alice-plugins

# 3. Push to GitHub
bash ../setup_github_repo.sh

# Or manually:
git init
git add .
git commit -m "Add interview-generated plugins"
gh repo create alice-plugins --public --source=. --push
```

### For End Users (Who Were Interviewed)

```bash
# Method 1: Claude Code UI
/plugin marketplace add [your-username]/alice-plugins
/plugin install automated-code-review@alice-plugins
/plugin install test-generation@alice-plugins

# Method 2: Quick install script
bash quick-install.sh [your-username]

# Method 3: Project configuration (.claude/settings.json)
{
  "plugins": {
    "automated-code-review": {
      "marketplace": "[username]/alice-plugins",
      "version": "1.0.0"
    }
  }
}
```

## Customization

### Customize Plugin Generation

Edit `synalinks/src/modules/interview_agent/plugin_packaging.py`:

```python
class CustomPlugin(PluginTemplate):
    """Your custom plugin type."""

    def generate_manifest(self):
        # Custom manifest generation
        manifest = super().generate_manifest()
        manifest["custom_field"] = "custom_value"
        return manifest

    def generate_custom_component(self):
        # Add custom components
        return "Custom content"
```

### Add New Plugin Types

```python
# In plugin_packaging.py

def _determine_plugin_type(self, requirement: Requirement) -> str:
    """Map requirement category to plugin type."""

    custom_mappings = {
        "my_category": "my_custom_type",
        "another_category": "another_type",
    }

    category_to_type = {
        "automation": "command",
        "coding": "skill",
        **custom_mappings,  # Add your mappings
    }

    return category_to_type.get(requirement.category, "basic")
```

### Custom Templates

Create template files in `examples/plugin_templates/`:

```
plugin_templates/
├── command_template.md
├── skill_template.md
├── mcp_server_template.js
└── manifest_template.json
```

Load and use in generator:

```python
def load_custom_template(template_name):
    template_path = Path("examples/plugin_templates") / template_name
    return template_path.read_text()
```

## Advanced Features

### 1. Multi-Requirement Plugins

Combine related requirements into single plugin:

```python
# Group related requirements
related_reqs = [
    req for req in prd.requirements.all_requirements
    if req.category == "testing"
]

# Generate single comprehensive plugin
comprehensive_plugin = plugin_generator.generate_comprehensive_plugin(
    requirements=related_reqs,
    plugin_name="testing-suite",
)
```

### 2. Plugin Dependencies

Add dependencies to `plugin.json`:

```json
{
  "name": "my-plugin",
  "dependencies": {
    "other-plugin": "^1.0.0"
  },
  "peerDependencies": {
    "base-utilities": ">=2.0.0"
  }
}
```

### 3. MCP Server with Real Implementation

Replace placeholder `server.js`:

```javascript
const { Server } = require('@modelcontextprotocol/sdk/server');
const { StdioServerTransport } = require('@modelcontextprotocol/sdk/server/stdio');

const server = new Server({
  name: 'advanced-data-server',
  version: '1.0.0'
});

// Add actual tools
server.setRequestHandler('tools/list', async () => ({
  tools: [
    {
      name: 'query_database',
      description: 'Query the database',
      inputSchema: {
        type: 'object',
        properties: {
          query: { type: 'string' }
        }
      }
    }
  ]
}));

server.setRequestHandler('tools/call', async (request) => {
  if (request.params.name === 'query_database') {
    // Implement actual database query
    const result = await queryDatabase(request.params.arguments.query);
    return { result };
  }
});

const transport = new StdioServerTransport();
server.connect(transport);
```

### 4. Hooks Integration

Add event hooks to plugins:

**In plugin.json:**
```json
{
  "name": "my-plugin",
  "hooks": {
    "SessionStart": {
      "command": "node",
      "args": ["${CLAUDE_PLUGIN_ROOT}/hooks/session-start.js"]
    },
    "PreToolUse": {
      "command": "python",
      "args": ["${CLAUDE_PLUGIN_ROOT}/hooks/pre-tool.py"]
    }
  }
}
```

## Testing Locally

Before pushing to GitHub, test locally:

```bash
# 1. Create local test marketplace
mkdir -p ~/.claude/local-marketplaces/test-marketplace
cp -r interview_output/marketplace/alice-plugins/* ~/.claude/local-marketplaces/test-marketplace/

# 2. Add local marketplace in Claude Code
/plugin marketplace add file://~/.claude/local-marketplaces/test-marketplace

# 3. Test plugin installation
/plugin install automated-code-review@test-marketplace

# 4. Test plugin functionality
/code_review
```

## Troubleshooting

### Plugin Not Found

**Issue:** `/plugin install` fails with "plugin not found"

**Solutions:**
- Check `.claude-plugin/marketplace.json` exists
- Verify `source` paths are correct (relative to marketplace.json)
- Ensure GitHub repo is public
- Check plugin names match exactly (case-sensitive)

### MCP Server Won't Start

**Issue:** MCP server plugin installed but server doesn't start

**Solutions:**
- Check `.mcp.json` syntax
- Verify `command` is installed (e.g., `node`, `python`)
- Check `${CLAUDE_PLUGIN_ROOT}` path is correct
- Look at Claude Code logs for errors
- Ensure server.js has no syntax errors

### Commands Not Appearing

**Issue:** Slash commands from plugin don't show up

**Solutions:**
- Verify `commands/` directory is at plugin root (not nested)
- Check `.md` files have correct YAML frontmatter
- Ensure plugin is enabled: `/plugin list --installed`
- Restart Claude Code
- Check file names are kebab-case

## Best Practices

### 1. Plugin Naming

- ✅ Use kebab-case: `my-plugin-name`
- ✅ Be descriptive: `automated-code-review` not `acr`
- ✅ Match category: `test-generation` for testing
- ❌ Avoid: spaces, underscores, camelCase

### 2. Documentation

- ✅ Comprehensive README with examples
- ✅ CLAUDE.md with context
- ✅ Clear command/skill descriptions
- ✅ List requirements and dependencies

### 3. Versioning

- ✅ Use semantic versioning: `1.0.0`, `1.1.0`, `2.0.0`
- ✅ Tag releases in git
- ✅ Document breaking changes
- ✅ Update version in plugin.json

### 4. Security

- ✅ Don't hardcode secrets
- ✅ Use environment variables
- ✅ Document required env vars
- ✅ Validate all inputs

## Integration with Interview Agent

### Automatic Flow

The complete integration:

```python
# 1. Interview
interview_result = await interview_agent(session)

# 2. Generate PRD
prd = await prd_generator(interview_result.session)

# 3. Generate plugins
plugin_generator = PluginGenerator()
plugins = plugin_generator.generate_bundle_from_prd(prd)

# 4. Create marketplace
marketplace_builder = MarketplaceBuilder()
marketplace = marketplace_builder.create_marketplace_repo(
    marketplace_name="interview-plugins",
    plugins_dir=plugins,
    prd=prd,
)

# 5. Push to GitHub (manual or automated)
```

### Workflow Summary

```
User Interview
      ↓
Requirements Discovery (5-10 requirements)
      ↓
PRD Generation (organized by category/priority)
      ↓
Plugin Generation (1 plugin per high-priority requirement)
      ↓
Marketplace Packaging (all plugins → marketplace.json)
      ↓
GitHub Push (automated script)
      ↓
Easy Installation (/plugin marketplace add ...)
```

## Examples

See `examples/generate_plugins_from_interview.py` for:
- Complete end-to-end flow
- Mock interview data
- Custom plugin generation
- Marketplace creation
- Installation automation

## Next Steps

1. **Run the generator**: `python examples/generate_plugins_from_interview.py`
2. **Review generated plugins**: Check `interview_output/plugins/`
3. **Test locally**: Use local marketplace
4. **Push to GitHub**: Run `setup_github_repo.sh`
5. **Share with users**: Give them installation commands

## Resources

- [Claude Code Plugin Docs](https://code.claude.com/docs/en/plugins)
- [MCP Specification](https://modelcontextprotocol.io/)
- [Plugin Marketplaces](https://code.claude.com/docs/en/plugin-marketplaces)
- [Example Plugins](https://github.com/anthropics/claude-code/tree/main/plugins)

---

**Generated by Interview Agent Plugin Packaging System**
