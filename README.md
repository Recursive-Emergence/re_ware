# RE_ware: Recursive Emergence Software Engineering Framework

**A principled, sensor-first architecture for intelligent software lifecycle management through externalized memory, structured reasoning, and safe evolution.**

## Mission

Enable agents to recursively improve software projects by implementing the Recursive Emergence (RE) triad—Ψ (externalized memory), Φ (coherence projection), and Ω (safety guardrails)—creating a reusable framework for systematic, methodical problem-solving guided by economic prudence and entropic discipline.

**Core Philosophy**: Intelligence emerges from the iterative cycle where fresh external memory (Ψ) enables accurate reasoning (Φ) within safety constraints (Ω), reducing contradiction and unlocking new capabilities over time.

## Architecture Overview

RE_ware implements a **sensor-first architecture** that ensures externalized memory is always fresh before any reasoning or decision-making occurs:

```
🔄 RE Pattern Flow:
Sensors → SensorHub → Ψ Update → Φ Projection → Ω Gates → ActionHub → External Reality
```

### The Recursive Emergence Triad

- **Ψ (Psi) - Externalized Memory**: Ontological graph of project knowledge maintained by continuous sensor input
- **Φ (Phi) - Coherence Projection**: AI-powered analysis with structured prompts, context budgeting, and entropy assessment  
- **Ω (Omega) - Safety Guardrails**: Executable constraints and validation gates that ensure safe evolution

### Key Components

1. **SensorHub** - Robust ingestion bus that normalizes events from all sources into the ontology
2. **Ontological Graph** - Gene/Phenotype architecture with NetworkX optimization for fast queries
3. **LLM Integration** - Structured JSON prompts with intelligent context selection and phi signals
4. **ActionHub** - Hybrid local/external action dispatcher with deduplication and traceability
5. **Project Agent** - AI assistant navigator providing context-aware guidance

## Quick Start

```python
from re_ware import create_project_intelligence, create_ontology_with_gene

# Create ontology with project management schema
graph = create_ontology_with_gene("project_manager")

# Set up project intelligence with LLM integration  
intelligence = create_project_intelligence(graph)

# Get AI context for current project state
context = await intelligence.query_project("What are the next actions needed?")

# Generate structured advice frame for decision making
advice = await intelligence.generate_advice_frame(
    species_id="project_manager_v1",
    instance_id="my_project",
    phi_state={"phi0": False}
)
```

## Core Architectural Benefits

- **🔄 Continuous Fresh State**: Ψ never stale, always reflects current reality through sensor input
- **🧠 Structured Intelligence**: JSON-based prompts with context budgeting and entropy analysis
- **🔗 Graph Optimization**: Fast neighbor queries and relationship traversal with NetworkX backing
- **📊 Real-time Metrics**: Live phi signals (coverage, entropy, criticality) for accurate reasoning
- **🛡️ Safe Evolution**: All changes validated against fresh project state and safety constraints
- **🤖 AI Assistant Ready**: Context-aware guidance for Claude Code, GitHub Copilot, and other tools

## Implementation Architecture

### Sensor-First Data Flow

1. **External Sources** (Git, Filesystem, GitHub, Human) generate events continuously
2. **Sensors** poll their sources and normalize events to standard format
3. **SensorHub** batches events (300ms windows) and applies mapping rules
4. **Ψ Memory** receives fresh, validated updates with watermark tracking
5. **Φ Projection** performs analysis based on current, accurate state
6. **Ω Gates** validate proposed actions for safety and coherence
7. **ActionHub** executes approved actions with deduplication and tracing
8. **Feedback Loop** creates new external reality that sensors detect

### Enhanced LLM Integration

RE_ware uses structured **AdviceInput JSON** prompts that include:

```json
{
  "identity": {"species_id": "project_manager_v1", "instance_id": "..."},
  "omega": {"dod_predicates": ["spec_has_tests"], "two_key_required": ["merge_main"]},
  "phi": {
    "phi0": false,
    "signals": {
      "coverage_ratio": 0.85,
      "entropy_hint": 0.3,
      "open_p0_issues": 1,
      "context_budget_used": 6
    }
  },
  "pulse": {"changed_nodes": ["req:auth"], "counters": {...}},
  "cards": [{"id": "req:auth", "type": "Requirement", ...}],
  "tools": {"github.issue": true, "fs.write": true}
}
```

The system returns structured advice with actions, tensions, and pull-more requests for optimal context management.

## Gene/Phenotype Architecture

The ontology uses a **Gene/Phenotype** pattern:

- **Gene**: Immutable schema templates defining node types, relationships, and validation rules
- **Phenotype**: Mutable runtime state with actual project data and NetworkX optimization
- **Validation**: All nodes and edges validated against gene constraints
- **Optimization**: NetworkX backing for O(1) neighbor queries and traversal

## Project Management Integration

### Schema-Driven Mapping

Configure project mapping via `sensors.yml`:

```yaml
rules:
  - match: "src/**/*.py"
    node: {type: CODEMODULE, fields: {language: "python"}}
    edges: [{rel: IMPLEMENTS, from: "{module}", to: "req:{infer}"}]
    
  - match: "tests/**/*test*.py"  
    node: {type: TEST, fields: {framework: "pytest"}}
    edges: [{rel: VERIFIES, from: "{test}", to: "code:{module}"}]
```

### Action Categories

- **Local Actions**: File writes, tests, formatting, graph updates (no external dependencies)
- **GitHub Actions**: Issues for collaborative work, PRs for features (proper collaboration)
- **Git Actions**: Branch creation, commits via local git commands

## Evolution & Self-Improvement

The system evolves through:

1. **Tension Detection**: Identifies contradictions and misalignments in the project state
2. **Coverage Analysis**: Tracks requirements without tests, APIs without docs
3. **Entropy Assessment**: Measures project complexity and risk factors
4. **Safe Actions**: Only proposes changes that pass Ω validation gates
5. **Learning**: Updates patterns and improves sensor accuracy over time

## Installation

```bash
# Clone repository
git clone https://github.com/your-org/re_software.git
cd re_software

# Install dependencies
pip install -r requirements.txt

# Optional: Configure LLM provider
export ANTHROPIC_API_KEY="your-key"  # or OPENAI_API_KEY
export LLM_PROVIDER="anthropic"      # or "openai"
```

## Key Files

- `ontology.py` - Ψ memory with Gene/Phenotype architecture and NetworkX optimization
- `llm_integration.py` - Φ projection with AdviceInput JSON prompts and context budgeting
- `sensor_hub.py` - Core ingestion bus with batch processing and watermarks
- `sensors.py` - Individual sensor implementations (Git, FS, GitHub, CLI)
- `action_hub.py` - Hybrid action dispatcher with Ω gating and traceability
- `project_agent.py` - AI assistant navigation and guidance
- `schemas/project_manager.json` - Comprehensive project management ontology

## License

This project implements Recursive Emergence principles for software engineering. The framework is designed to be:

- **Reusable** across millions of future software projects
- **Principled** through consistent RE triad implementation  
- **Safe** via executable guardrails and validation gates
- **Intelligent** through structured LLM integration and context optimization

---

*RE_ware represents a new paradigm in software lifecycle management: where intelligence emerges from principled externalized memory, structured reasoning, and safe evolutionary processes.*