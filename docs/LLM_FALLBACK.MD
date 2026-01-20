# LLM Provider Fallback Strategy

Spectrue Engine implements a provider fallback mechanism for **claim extraction** when the primary LLM provider (DeepSeek) experiences execution failures.

## Overview

The fallback system is designed to handle **execution failures** (connectivity, timeouts, server errors) by seamlessly switching to a secondary provider (OpenAI GPT-5.2). It does **not** handle semantic failures (low quality content).

**Primary Provider:** `deepseek-chat` (Type: DeepSeek)
**Fallback Provider:** `gpt-5.2` (Type: OpenAI)

## Failure Classification

Failures are classified into the following types (`spectrue_core.llm.failures.LLMFailureKind`):

| Failure Type | Description | Keywords |
|--------------|-------------|----------|
| `CONNECTION_ERROR` | Network/socket issues | `connection`, `network`, `refused` |
| `TIMEOUT` | Request timed out | `timeout`, `timed out`, `deadline exceeded` |
| `PROVIDER_ERROR` | Upstream provider error | `rate limit`, `503`, `overloaded` |
| `INVALID_JSON` | Response not valid JSON | `json`, `parse`, `expecting value` |
| `SCHEMA_VALIDATION_FAILED` | JSON violates schema | `validation failed`, `missing` |

## Fallback Behavior

1. **Attempt Primary:** The system attempts the operation with the Primary Provider.
2. **Detect Failure:** If an exception occurs, it is classified against the rules above.
3. **Trace Failure:** A trace event `llm.call.failed` is emitted.
4. **Switch:** If eligible, the system switches to the Fallback Provider (`llm.fallback.used`).
5. **Attempt Fallback:** The operation is retried with the Fallback Provider.
6. **Final Result:**
   - If successful: Returns the fallback result (trace marks `is_fallback=True`).
   - If failure: A `llm.fallback.failed` event is emitted.

## Configuration

Fallback models are configured in `EngineRuntimeConfig`:

```python
model_claim_extraction_fallback: str = ModelID.PRO  # "gpt-5.2"
```

Override via ENV: `MODEL_CLAIM_EXTRACTION_FALLBACK="gpt-4o"`
