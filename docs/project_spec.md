# RE_ware Core — Spec v0.1 (RE-first, concise)

## Mission
Evolve into a reusable, general agent that manages any project's lifecycle by reusing human engineering practice, under Recursive Emergence (RE): Ψ = externalized project memory, Ω = executable guardrails, Φ = readiness/coherence snapshots.

## Agent Gene (immutable at birth)
species_id, version, purpose, protocols=[announce, request, propose, commit, verify, critique, error].
Persist {species_id, instance_id} to graph + sidecar file for self/other recognition and restart continuity.

## Memory Model
- **Hot (RAM):** ring buffer of GraphPulse frames (recent deltas, counters, last events) + 5–8 node LLM cards.
- **Warm (file):** JSONL snapshots of compressed cards/deltas for instant reload.
- **Cold (repo):** artifacts (code/docs/tests/build logs).

## Minimal Ontology (Ψ)
Nodes: Problem, Spec, Requirement, Plan, Task, Test, CodeModule, Issue, Build, Decision, Pattern, Budget.  
Rels: REFINES, IMPACTS, IMPLEMENTS, VERIFIES, COVERS, DEPENDS_ON, SUPERSEDES, ALLOCATES.  
State on every node: {status, version, last_changed, change_summary, criticality, owners}.

## Guardrails (Ω)
- DoD: every Requirement impacted by a Spec is verified by ≥1 Test; no open P0 Issues.
- Two-key transitions: merge_main, release_prod.
- Scoped capabilities: filesystem sandbox; network off by default.

## Projection (Φ)
- **Φ₀ (boolean):** DoD holds ∧ two-key gates satisfied.
- **Φ₁ (raw):** {req_total, tests_total, uncovered, coverage_ratio, recent_changes_7d}.

## LLM Interface (compressed + pull-more)
- LLM consumes ≤8 cards + 1 GraphPulse frame; may reply "need more: [node_ids|paths]".
- Citations always to node ids + versions; responses remain deterministic JSON where possible.

## Loop (reflect → plan → act → verify → learn)
- Reflect: compute GraphPulse + Φ.
- Plan: derive minimal Tasks to close uncovered gaps.
- Act: execute through tool adapters; write edges and states to Ψ.
- Verify: run tests; recompute Φ; apply Ω gates.
- Learn: mint Pattern(s) from success; update Decisions; snapshot warm state.
- Release when Φ₀ true; continue if new Problems/Requirements arrive.

## Reproduction (adaptive, non-numeric)
Spawn scoped offspring only when: backlog_pressure>baseline ∧ resources_available ∧ Ω permits. Children inherit a subset gene + stricter Ω and must publish traces; TTL-bounded.

## API (RE_ware CLI)
```bash
# Show project consciousness state  
python evolve.py status

# Get project reasoning and advice
python evolve.py advice

# Execute single evolution cycle
python evolve.py tick

# Enable autonomous project management  
python evolve.py auto --interval 5
```

## Success

One-line start; autonomous ontology growth from repo; continuous intelligence; contradiction detection prevents drift; releases gated by Ω; traces produce reusable Patterns.

## Test Change  
Testing git change detection and compression into ontological nodes.

Second test to verify git tracker picks up new changes.
## Dynamic Test
Added during agent run.
