# 4. Provider-agnostic LLM client

- **Status:** Accepted
- **Date:** 2026-01-25

## Context

The generation stage must call a chat LLM, and we want to demonstrate the system
against more than one vendor (Anthropic and OpenAI) without scattering SDK calls
through the codebase. "Provider parity" is a stated design principle: anything
that works with one provider should work with the other. Leaking a vendor SDK
into the pipeline, prompts, or UI would make that principle impossible to hold
and would couple unrelated modules to a specific API shape.

## Decision

All LLM access goes through a minimal `LLMClient` protocol with a single method,
`complete(system, user) -> CompletionResult`. `CompletionResult` is a normalized
dataclass (text, input/output tokens, latency, resolved model name) so callers
never see a vendor-specific response object. Two implementations,
`AnthropicClient` and `OpenAIClient`, are selected by `get_llm_client` from
`settings.llm_provider`. No other module imports a provider SDK. See
`generation/llm_client.py`.

SDKs are imported lazily inside each client's constructor so that selecting one
provider does not require the other's package to be installed, and constructors
accept an optional pre-built client for testing without network access.

## Consequences

- The pipeline, prompts, and UI are written once and run against either provider.
- Token counts and latency are captured uniformly into the `QueryTrace`,
  regardless of vendor.
- Adding a third provider is a new `LLMClient` implementation plus one branch in
  `get_llm_client` — no changes elsewhere.
- The interface is intentionally minimal (single-shot completion). Streaming and
  tool use are not part of the protocol today; adding them would extend the
  protocol and is tracked in the roadmap.
