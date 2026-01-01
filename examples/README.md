# Examples

This directory contains example scripts demonstrating how to use Spectrue Engine.

## Examples

### 1. Basic Usage (`basic_usage.py`)

The simplest example showing how to analyze a text claim.

```bash
# Set your API keys
export OPENAI_API_KEY="sk-..."
export TAVILY_API_KEY="tvly-..."

# Run the example
python examples/basic_usage.py
```

### 2. Claim Orchestration (`claim_orchestration.py`)

Advanced example demonstrating claim-centric orchestration:
- Building execution plans based on claim metadata
- Progressive widening with phase runner
- Evidence sufficiency checking
- Weighted RGBA aggregation

```bash
python examples/claim_orchestration.py
```

### 3. Posterior Calibration (`calibrate_claim_posterior.py`)

Fit alpha/beta for the claim posterior model using labeled data (MAP estimation).

```bash
python examples/calibrate_claim_posterior.py --input path/to/claims.jsonl
```

## Running Examples

1. Install the package in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

2. Set required environment variables:
   ```bash
   export OPENAI_API_KEY="your-key"
   export TAVILY_API_KEY="your-key"
   ```

3. Run any example:
   ```bash
   python examples/<example_name>.py
   ```

## Creating Your Own

Use these examples as templates. Key imports:

```python
# Core engine
from spectrue_core.engine import SpectrueEngine
from spectrue_core.config import SpectrueConfig

# Orchestration
from spectrue_core.verification.orchestrator import ClaimOrchestrator
from spectrue_core.verification.phase_runner import PhaseRunner
from spectrue_core.verification.execution_plan import BudgetClass
from spectrue_core.verification.sufficiency import evidence_sufficiency

# Claim Metadata
from spectrue_core.schema.claim_metadata import (
    ClaimMetadata,
    VerificationTarget,
    ClaimRole,
)
```
