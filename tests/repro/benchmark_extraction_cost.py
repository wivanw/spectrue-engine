from dataclasses import dataclass

@dataclass
class ModelPrice:
    input_usd_per_1m: float
    output_usd_per_1m: float

# Pricing from default_pricing.json
PRICING = {
    "deepseek-chat": ModelPrice(0.14, 0.28),
    "gpt-5-nano": ModelPrice(0.05, 0.40), # 0.00000005 * 1M = 0.05
    "gpt-5": ModelPrice(1.25, 10.00)
}

def estimate_tokens(text: str) -> int:
    return len(text) // 4

def calculate_cost(model: str, input_tok: int, output_tok: int) -> float:
    p = PRICING[model]
    return (input_tok * p.input_usd_per_1m / 1e6) + (output_tok * p.output_usd_per_1m / 1e6)

# Scenario
CHUNK_TEXT_LEN = 4000 # ~1000 tokens
NUM_CLAIMS = 5

# --- Monolithic (DeepSeek Only) ---
mono_prompt_len = CHUNK_TEXT_LEN + 2000 # Instructions are heavy
mono_output_len = NUM_CLAIMS * 500 # Full metadata is heavy
mono_input_tok = estimate_tokens("a" * mono_prompt_len)
mono_output_tok = estimate_tokens("a" * mono_output_len)

cost_mono = calculate_cost("deepseek-chat", mono_input_tok, mono_output_tok)

# --- Split (DeepSeek Core + Nano Enrichment) ---

# 1. Core (DeepSeek)
core_prompt_len = CHUNK_TEXT_LEN + 500 # Light instructions
core_output_len = NUM_CLAIMS * 100 # Only text + normalized
core_input_tok = estimate_tokens("a" * core_prompt_len)
core_output_tok = estimate_tokens("a" * core_output_len)

cost_core = calculate_cost("deepseek-chat", core_input_tok, core_output_tok)

# 2. Enrichment (Nano)
# Repetitive context: Pass chunk text again + core claim text
enrich_prompt_len_per_claim = CHUNK_TEXT_LEN + 500 # Context + specific instructions
enrich_output_len_per_claim = 400 # Metadata without core text

enrich_input_tok_total = NUM_CLAIMS * estimate_tokens("a" * enrich_prompt_len_per_claim)
enrich_output_tok_total = NUM_CLAIMS * estimate_tokens("a" * enrich_output_len_per_claim)

cost_enrich = calculate_cost("gpt-5-nano", enrich_input_tok_total, enrich_output_tok_total)

total_split = cost_core + cost_enrich

print(f"--- Cost Benchmark (Chunk={CHUNK_TEXT_LEN} chars, Claims={NUM_CLAIMS}) ---")
print(f"Monolithic (DeepSeek): ${cost_mono:.6f}")
print(f"  Input: {mono_input_tok} tok, Output: {mono_output_tok} tok")
print("")
print(f"Split Pipeline:        ${total_split:.6f}")
print(f"  Core (DeepSeek):     ${cost_core:.6f} (In: {core_input_tok}, Out: {core_output_tok})")
print(f"  Enrich (Nano):       ${cost_enrich:.6f} (In: {enrich_input_tok_total}, Out: {enrich_output_tok_total})")
print("")
print(f"Difference:            ${total_split - cost_mono:.6f}")
print(f"Ratio (Split/Mono):    {total_split/cost_mono:.2f}x")

if total_split < cost_mono:
    print("RESULT: SAVINGS")
else:
    print("RESULT: COST INCREASE")
