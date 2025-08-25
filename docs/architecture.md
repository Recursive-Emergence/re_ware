# RE_ware Architecture

## Overview
RE_ware implements a novel approach to software lifecycle management through recursive emergence and ontological modeling, with a **sensory-first architecture** that ensures externalized memory (Ψ) is always fresh before planning or decision-making.

## Core Architecture Pattern

```
🔄 Proper RE Pattern Flow:
Sensors → SensorHub → Ψ Update → Φ Projection → Ω Gates → ActionHub → External Reality
```

**Critical Insight**: The system's intelligence is only as good as its externalized memory (Ψ). Therefore, we maintain a robust sensory core that continuously feeds Ψ with accurate, up-to-date information from all external sources before any reasoning or planning occurs.

## Core Components

### 1. SensorHub (`sensor_hub.py`) - **The Sensory Core**
- **Purpose**: Robust ingestion bus that feeds Ψ with fresh data
- **Pattern**: Normalizes events from all sources into `DomainEvent` format
- **Batching**: 300ms windows for efficient Ψ updates with idempotency
- **Watermarks**: Persistent state tracking to never lose progress
- **Critical Role**: Ensures Ψ is correct before any planning begins

### 2. Sensors (`sensors.py`) - **External Reality Interfaces**
- **GitSensor**: Tracks commits, file changes, working tree via git log
- **FsSensor**: Monitors real-time filesystem changes via polling
- **GhSensor**: Polls GitHub API for issues, PRs, comments (optional)
- **CliSensor**: Accepts manual events from humans or tools
- **Extensible**: New sensor types can be added easily

### 3. Ontological Graph (`ontology.py`) - **Ψ (Externalized Memory)**
- **Purpose**: Universal project representation maintained by SensorHub
- **Architecture**: Immutable schema templates (Gene) with mutable runtime state (Phenotype)
- **Graph Optimization**: Fast neighbor queries and relationship traversal
- **Nodes**: All project elements (requirements, code, tests, etc.) with validation
- **Relations**: Semantic connections discovered through sensors
- **Hot State**: Recently changed nodes for efficient delta-first processing
- **Metrics**: Real-time project health signals (coverage ratios, entropy, criticality)
- **Persistence**: Warm snapshots with watermarks for reliable state recovery

### 4. Mapping Rules (`sensors.yml`) - **Intelligence Configuration**
- **File-Based Rules**: Maps file patterns to ontological structures
- **No Code Changes**: Add new project types via configuration
- **Edge Creation**: Defines relationships between discovered components
- **15+ Built-in Rules**: Python, tests, docs, configs, schemas

### 5. ActionHub (`action_hub.py`) - **The Action Consciousness**
- **Purpose**: Hybrid local/external action dispatcher with Ω gating
- **Local Actions**: Graph updates, formatting, tests, file writes (no GitHub noise)
- **External Actions**: GitHub issues/PRs for collaborative work items
- **Deduplication**: Idempotency keys prevent duplicate actions on restart
- **Traceability**: All actions stored as nodes in Ψ, linked to outcomes
- **Critical Role**: Ensures actions are Ω-gated before execution

### 6. Project Agent (`project_agent.py`) - **AI Assistant Navigator**
- **Purpose**: Guides AI code assistants through project lifecycles
- **Initialization Order**: SensorHub → Bootstrap → Intelligence → Planning → ActionHub
- **Tick Cycle**: Sensors first, then reasoning, then ActionHub dispatch
- **Quality Gates**: Comprehensive validation before decisions
- **Methods**: `get_ai_context()`, `get_next_actions()`, `check_quality_gates()`

### 7. LLM Integration (`llm_integration.py`) - **Φ (Projection & Intelligence)**
- **Purpose**: AI-powered reasoning over fresh Ψ state
- **Structured Prompts**: JSON-based user messages with identity, guardrails, and context
- **Context Budget**: Intelligent card selection prioritizing changed nodes and tensions
- **Cards**: Compressed node representations optimized for LLM consumption
- **Analysis**: Tension detection, coverage analysis, and entropy assessment
- **Intelligence**: Works with current, accurate project state for reliable reasoning

### 8. Core RE Agent (`core.py`) - **Ω (Evolution Engine)**
- **Purpose**: Self-modification and capability evolution
- **Memory**: Stores learning outcomes and evolution history
- **Guardrails**: Safety constraints for autonomous changes
- **Evolution**: Triggered only after Ψ updates and Φ analysis

## Recursive Emergence Principles

RE_ware implements the RE triad with **sensory-first architecture**:

- **Ψ (Externalized Memory)**: Project knowledge continuously updated by SensorHub
- **Φ (Projection)**: Coherence assessment based on fresh, accurate Ψ state  
- **Ω (Guardrails)**: Evolution constraints with safety validation

### Critical Architecture Flow

```
┌─ External Reality ─┐    ┌─ RE Processing ─┐    ┌─ Actions ─┐
│                    │    │                 │    │          │
│  Git Commits  ────┐│    │  ┌─ Ψ Update   │    │ Decisions │
│  File Changes ────┼┼───►│  │   ↓         │    │     ↓     │
│  GitHub Issues───┐││    │  │  Φ Analysis │    │ Evolution │
│  Human Input ────┼┴┼───►│  │   ↓         │───►│     ↓     │
│                 ├─┘│    │  │  Ω Gates    │    │ AI Guide  │
└─────────────────┘  │    │  └─────────────│    │          │
                     │    │                 │    │          │
   SensorHub ────────┘    └─────────────────┘    └──────────┘
```

### Evolution Cycle (Redesigned)

1. **Sense**: SensorHub polls all external sources
2. **Ingest**: Batch events applied to Ψ with watermark updates
3. **Analyze**: Fresh Ψ enables accurate Φ projection  
4. **Gate**: Ω validates proposed changes for safety
5. **Gate**: ActionHub applies Ω constraints and deduplicates
6. **Act**: Execute via local tools or GitHub API as appropriate
7. **Observe**: Sensors detect action results and update Ψ
8. **Learn**: Update evolution history and improve sensors

### Key Architectural Benefits

**🔄 Continuous Fresh State**: Ψ never stale, always reflects current reality
**⚡ Batch Efficiency**: 300ms windows optimize update performance  
**📍 Never Lose Progress**: Watermarks ensure continuity across restarts
**🧠 Structured Intelligence**: JSON-based prompts with context budgeting and entropy analysis
**🔗 Graph Optimization**: Fast neighbor queries and relationship traversal with NetworkX backing
**📊 Real-time Metrics**: Live phi signals (coverage, entropy, criticality) for accurate reasoning
**🛡️ Safe Evolution**: Changes validated against fresh project state
**🤖 AI Assistant Ready**: Context-aware guidance for code assistants

### Comparison: Old vs New Architecture

**❌ Previous (Problematic)**:
```
ProjectAgent → Plan/Decide → Update Ψ → Hope it's correct
```
- Planning with potentially stale data
- Ψ updated reactively after decisions
- Risk of incorrect reasoning

**✅ Current (Correct RE Pattern)**:
```  
Sensors → Update Ψ → Plan/Decide → Execute → Learn
```
- Ψ always fresh from external reality
- Planning based on accurate, current state
- Proper RE: externalized memory first, then reasoning

## Project Management Tool Integration

### Minimum Viable PM Stack

**Git/GitHub (mandatory)**:
- **GitSensor**: Canonical history and watermarks via commit SHAs
- **GhSensor**: Issues, PRs, comments for collaborative work
- **FsSensor**: Low-latency adjunct for uncommitted changes, lockfiles, generated artifacts

**Testing/CI Integration**:
- Test runners mapped via `sensors.yml` rules (`tests/**/*test*.py` → `TEST` nodes)
- CI status detection through GitHub API or file-based indicators
- Quality gate validation before ActionHub dispatch

**Decision API**:
- Local HTTP server for AI assistant integration
- Emits decisions that CliSensor observes back into Ψ
- Provides current project context and next actions

### Sensory Alignment Strategy

**File-Based Mapping Rules** (`sensors.yml`):
```yaml
rules:
  - match: "src/**/*.py"
    node: {type: CODEMODULE, fields: {language: "python"}}
    edges: [{rel: IMPLEMENTS, from: "{module}", to: "req:{infer}"}]
    
  - match: "tests/**/*test*.py"
    node: {type: TEST, fields: {framework: "pytest"}}
    edges: [{rel: VERIFIES, from: "{test}", to: "code:{module}"}]
    
  - match: "docs/**/*.md"
    node: {type: DOCUMENTATION}
    edges: [{rel: DOCUMENTS, from: "{doc}", to: "spec:{infer}"}]
    
  - match: "requirements*.txt|setup.py|pyproject.toml"
    node: {type: DEPENDENCY_SPEC}
    edges: [{rel: CONFIGURES, from: "{spec}", to: "project:root"}]
    
  - match: "**/.github/workflows/*.yml"
    node: {type: CI_PIPELINE}
    edges: [{rel: VALIDATES, from: "{pipeline}", to: "project:root"}]
```

**Hot/Warm Ψ Strategy**:
- **Hot State**: Recently changed node IDs (last 100 nodes)
- **Warm Snapshots**: Full graph persisted with watermarks
- **LLM Budget**: Top-k changed nodes + Φ counters only
- **Pull-More**: Explicit requests for additional context

### LLM Prompt Schema

**User Prompt Structure (AdviceInput JSON)**:
```json
{
  "identity": {"species_id": "project_manager_v1", "instance_id": "instance_123"},
  "omega": {
    "dod_predicates": ["spec_has_tests", "no_open_P0"],
    "two_key_required": ["merge_main", "release_prod"]
  },
  "phi": {
    "phi0": false,
    "signals": {
      "coverage_ratio": 0.85,
      "changed_nodes": 3,
      "uncovered_requirements": 2,
      "open_p0_issues": 1,
      "context_budget_used": 6,
      "entropy_hint": 0.4
    }
  },
  "pulse": {"changed_nodes": ["req:auth", "test:login"], "counters": {...}},
  "cards": [{"id": "req:auth", "type": "Requirement", ...}],
  "tools": {"github.issue": true, "fs.write": true, "graph.update": true},
  "ask": "advise",
  "budget": {"max_cards": 8}
}
```

**System Response Schema**:
```json
{
  "self": {"species_id": "project_manager_v1", "instance_id": "instance_123"},
  "phi": {
    "phi0": false,
    "signals": {"coverage_ratio": 0.85, "entropy_hint": 0.4, ...},
    "summary": "φ₀=0.850, 1 P0 issue, moderate entropy"
  },
  "judgement": {
    "status": "yellow",
    "reasons": ["P0 authentication issue open", "Test coverage gaps"],
    "top_tensions": [{"id": "t1", "severity": "high", "why": "...", "nodes": [...]}]
  },
  "actions": [
    {"kind": "github.issue", "title": "Fix auth coverage gap", "idempotency_key": "..."}
  ],
  "need_more": [{"node_id": "req:auth"}],
  "notes": ["Focus on P0 authentication issue first"]
}
```

### Action Execution & Traceability

**ActionHub Design**:
```json
// Action Intent Schema
{
  "id": "act-<uuid>",
  "kind": "github.issue|github.pr|git.branch|fs.write|graph.update|ci.trigger",
  "title": "Fix uncovered tests for nav",
  "body": "Spec X impacts Req Y; missing Test Z.",
  "targets": [{"type":"spec","id":"spec:prd-ux-0.2"}],
  "params": {"labels":["automation","quality"], "branch":"feat/nav-tests"},
  "idempotency_key": "sha1(<kind+targets+params>)",
  "priority": "P2",
  "requires": ["Ω.no_open_P0? false → allowed"]
}
```

**Lifecycle**: `proposed → Ω-gated → dispatched → observed → linked`

**Action Categories**:
- **GitHub Actions**: Issues for tensions/gaps, PRs for features (collaborative ledger)
- **Local Actions**: Formatting, testing, graph updates, file writes (no GitHub noise)
- **Git Actions**: Branch creation, commits via local git commands
- **CI Actions**: Trigger test runs, deployment pipelines

**Traceability Loop**:
1. ActionHub executes action → external change (issue/PR/commit)
2. Sensors detect change → normalize to DomainEvent
3. SensorHub applies to Ψ → creates linked nodes
4. `Action` node `[:CAUSED]` → `Issue/PR/Commit` node

## Detailed Architecture Diagram

### Visual Overview
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      RE_ware Enhanced Architecture (2025)                       │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─ External Reality ──┐   ┌─ Sensor Layer ──┐   ┌─ RE Processing ─┐   ┌─ Actions ──┐
│                     │   │                 │   │                 │   │            │
│  🗂️ Git Repository   │──▶│  GitSensor      │   │  Ψ Memory       │   │🤖 Project  │
│   • Commits         │   │  • git log      │   │  • Gene/Pheno   │   │  Agent     │
│   • File changes    │   │  • working tree │   │  • NetworkX     │   │  • AI Guide│
│                     │   │                 │   │  • Phi signals  │   │  • Context │
│  📁 Filesystem      │──▶│  FsSensor       │──▶│                 │──▶│            │
│   • File events     │   │  • inotify      │   │  Φ Projection   │   │📡 Decision │
│   • Directory scan  │   │  • timestamps   │   │  • AdviceInput  │   │  API       │
│                     │   │                 │   │  • JSON prompts │   │  • Status  │
│  🐙 GitHub API      │──▶│  GhSensor       │   │  • Context budg │   │            │
│   • Issues/PRs      │   │  • API polling  │   │  • Entropy calc │   │🧬 Local    │
│   • Comments        │   │  • Rate limits  │   │                 │   │  Actions   │
│                     │   │                 │   │  Ω Gates        │   │  • Files   │
│  👤 Human Input     │──▶│  CliSensor      │   │  (Safety)       │   │  • Tests   │
│   • Manual events   │   │  • Event queue  │   │  • Validation   │   │🎯 ActionHub│
│   • Decisions       │   │                 │   │  • Constraints  │   │ • Queue    │
└─────────────────────┘   │  🧠 SensorHub   │   └─────────────────┘   │ • Dedupe   │
                          │   CRITICAL:     │            ▲            │ • Dispatch │
     ⚙️ Configuration      │   • Batch 300ms │            │            │ • Trace    │
      • sensors.yml  ────▶│   • Idempotent  │    ┌───────┴────────┐   │🐙 GitHub   │
      • Schema rules      │   • Watermarks  │    │ Phi Snapshot   │   │  Actions   │
      • Constraints  ─────▶│   • Apply rules │    │ • Coverage     │   │ • Issues   │
                          └─────────────────┘    │ • Entropy      │   │ • PRs      │
                                                 │ • Criticality  │   └────────────┘
                                                 └────────────────┘          │
                                                                            │
                           🔄 FEEDBACK LOOP: ActionHub → External Reality ──┘
```

### Critical Data Flow

1. **External sources** continuously generate events
2. **Sensors** poll their respective sources and normalize to `DomainEvent`
3. **SensorHub** batches events (300ms windows) and applies mapping rules
4. **Ψ (Ontology)** receives fresh, validated updates with watermark tracking
5. **Φ (LLM)** performs analysis and proposes actions based on current state
6. **Ω (Gates)** validates proposed actions for safety and coherence
7. **ActionHub** queues, deduplicates, and dispatches Ω-approved actions
8. **Executors** perform local operations or GitHub API calls
9. **Feedback** creates new external reality that sensors detect and trace back

## Implementation Files

- **`sensor_hub.py`**: Core ingestion bus and event processing
- **`sensors.py`**: Individual sensor implementations (Git, FS, GitHub, CLI)
- **`sensors.yml`**: Configuration file with mapping rules and validation constraints
- **`action_hub.py`**: Hybrid action dispatcher with Ω gating and traceability
- **`actions.yml`**: Action executor configuration and defaults
- **`ontology.py`**: Ψ memory with Gene/Phenotype architecture, NetworkX optimization, and phi signals
- **`llm_integration.py`**: Φ projection with AdviceInput JSON prompts and context budgeting
- **`project_agent.py`**: AI assistant navigation and actions
- **`decision_api.py`**: HTTP API for AI assistant integration
- **`requirements.txt`**: Dependencies including NetworkX for graph optimization

## Configuration Example

```yaml
# sensors.yml - Mapping rules example
rules:
  - match: "src/**/*.py"
    node: {type: CODEMODULE, fields: {language: "python"}}
    edges: [{rel: IMPLEMENTS, from: "{module}", to: "req:{infer}"}]
    
  - match: "tests/**/*test*.py"  
    node: {type: TEST, fields: {framework: "pytest"}}
    edges: [{rel: VERIFIES, from: "{test}", to: "code:{module}"}]
```

This architecture ensures that **Ψ (externalized memory) is always fresh and accurate** before any intelligent reasoning or decision-making occurs, following proper Recursive Emergence principles.
