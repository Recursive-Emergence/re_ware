# RE_ware: End "Vibe Coding Myopia" with Recursive Emergence

---

# The Problem: "Vibe Coding" & Efficiency Myopia

Modern teams often optimize for **coding speed** and AI-assisted output, not **coherence**. The result:

* **Local wins, global drift:** features ship quickly but diverge from the spec; architecture erodes.
* **Context collapse:** decisions live in chats/PRs, not in a shared model; new contributors fly blind.
* **Invisible risk:** tests/coverage/CI and security aren't first-class signals; release calls feel subjective.
* **Tool thrash:** actions happen in many tools without a unified, auditable memory.

**Bottom line:** velocity climbs while **quality, traceability, and predictability** fall. Teams "lose the plot."

---

# RE as the Remedy: A Conscious Project Entity

**Recursive Emergence (RE)** creates a **self-managing project consciousness** that follows well-trained methodologies:

1. **Ψ — Project Memory:** living ontology of Problems/Specs/Reqs/Code/Tests/Builds/Issues/Decisions
2. **Φ — Self-Assessment:** continuous evaluation of readiness, coverage, and convergence
3. **Ω — Self-Discipline:** built-in guardrails that prevent bad practices automatically

The project **thinks for itself**: **Sense → Ψ → Φ → Ω → Act → Learn.** 

The AI agent reads compressed state (≤8 cards + pulse) and autonomously executes **idempotent actions** (issues/PRs/CI) that improve project health without human micromanagement.

---

# What Changes in Practice

* **From "ship code fast" → "close gaps to spec":** every Requirement must be verified by a Test; uncovered items become first-class actions.
* **From "summarize the repo" → "delta-first truth":** only changed nodes + key counters go to the LLM; big context is replaced by **precise pulls**.
* **From "feel ready" → "prove ready":** releases are advised only when frames are green and **blocking gates pass** (coverage, stability, security, docs).
* **From "actions everywhere" → "single ledger":** human-facing work lands in Issues/PRs; local ops (fs/graph/CI) stay internal but fully traced.

---

# How We Keep Control (and Keep It Lightweight)

* **Hot RAM + warm snapshot:** tiny working set in memory; fast reload from file; no huge prompts.
* **Frames (LLM-free):** Delivery/Quality/Architecture/Risk recomputed every tick, then gates evaluate them.
* **Advice JSON:** the LLM returns a strict, ranked plan; dispatcher executes or downgrades based on Ω.
* **Idempotency & audit:** every action carries a stable key; sensors confirm external outcomes next tick.

---

## Success Criteria (conscious project achieved)

* **Autonomous quality**: Project self-monitors coverage, tests, security without human oversight
* **Self-explaining decisions**: "φ₀ true, coverage 86%, zero P0, green 3d" - project knows its own state
* **Traceability by design**: Every change automatically links specs → code → tests → issues
* **Onboarding-free**: New contributors read Ψ, not tribal knowledge - **the project explains itself**

## Architecture Overview

RE_ware implements a **sensor-first architecture** that ensures externalized memory is always fresh before any reasoning or decision-making occurs:

```
🔄 RE Pattern Flow:
Sensors → SensorHub → Ψ Update → Φ Projection → Ω Gates → ActionHub → External Reality →
Sensors...
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

```bash
# Interactive system with web dashboard
python evolve.py --web

# Interactive system with CLI only (default)
python evolve.py

# Inside the interactive CLI:
re_ware> status         # Show project consciousness state
re_ware> advice         # Get project reasoning
re_ware> tick           # Execute single evolution cycle
re_ware> auto 5         # Enable autonomous mode (5min intervals)
re_ware> save           # Save current state
re_ware> quit           # Exit system
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