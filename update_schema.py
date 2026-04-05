import re

schema_file = "spectrue_core/agents/llm_schemas.py"
with open(schema_file, "r") as f:
    content = f.read()

# Remove expert_summary from CLAIM_JUDGE_SCHEMA required properties
content = re.sub(
    r'(\s*)"simple_summary",\s*"expert_summary",',
    r'\1"explanation",',
    content
)

# Remove expert_summary from CLAIM_JUDGE_SCHEMA properties
# The user's snippet shows:
#        "explanation": {
#            "type": "string",
#            "description": "Human-readable explanation of the verdict",
#        },
#        },
#        "expert_summary": {
# Wait, the user's snippet was a diff they tried to apply. Let's look at the actual file contents (lines 790-850)
