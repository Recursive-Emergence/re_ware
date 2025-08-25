# Evolution Engine Requirements

## REQ-EVE-001: State Management
The evolution engine SHALL maintain phi state including phi0, coherence, and stability metrics.

## REQ-EVE-002: Advice Caching  
The evolution engine SHALL cache advice responses to avoid redundant LLM calls when system state is unchanged.

## REQ-EVE-003: Interactive Commands
The evolution engine SHALL provide interactive commands for tick, advice, status, and save operations.

## REQ-EVE-004: Agent Initialization
The evolution engine SHALL initialize ontology, sensor hub, and RE agent components in proper sequence.

## REQ-EVE-005: Error Handling
The evolution engine SHALL handle initialization errors gracefully and provide clear error messages.