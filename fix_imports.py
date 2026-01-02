#!/usr/bin/env python3
"""
Fix imports after M119 verification/ reorganization.
This script updates all import statements to reflect the new subdirectory structure.
"""

import os
import re
from pathlib import Path

# Mapping of old imports to new imports
IMPORT_MAPPINGS = {
    # Search subdirectory
    'from spectrue_core.verification.search_mgr import': 'from spectrue_core.verification.search.search_mgr import',
    'from spectrue_core.verification.search_policy import': 'from spectrue_core.verification.search.search_policy import',
    'from spectrue_core.verification.search_policy_adapter import': 'from spectrue_core.verification.search.search_policy_adapter import',
    'from spectrue_core.verification.source_utils import': 'from spectrue_core.verification.search.source_utils import',
    'from spectrue_core.verification.retrieval_eval import': 'from spectrue_core.verification.search.retrieval_eval import',
    'from spectrue_core.verification.retrieval_trace import': 'from spectrue_core.verification.search.retrieval_trace import',
    'from spectrue_core.verification.trusted_sources import': 'from spectrue_core.verification.search.trusted_sources import',
    
    # Calibration subdirectory
    'from spectrue_core.verification.calibration_registry import': 'from spectrue_core.verification.calibration.calibration_registry import',
    'from spectrue_core.verification.calibration_models import': 'from spectrue_core.verification.calibration.calibration_models import',
    
    # Orchestration subdirectory
    'from spectrue_core.verification.orchestrator import': 'from spectrue_core.verification.orchestration.orchestrator import',
    'from spectrue_core.verification.phase_runner import': 'from spectrue_core.verification.orchestration.phase_runner import',
    'from spectrue_core.verification.execution_plan import': 'from spectrue_core.verification.orchestration.execution_plan import',
    'from spectrue_core.verification.stop_decision import': 'from spectrue_core.verification.orchestration.stop_decision import',
    'from spectrue_core.verification.sufficiency import': 'from spectrue_core.verification.orchestration.sufficiency import',
    
    # Claims subdirectory
    'from spectrue_core.verification.claim_selection import': 'from spectrue_core.verification.claims.claim_selection import',
    'from spectrue_core.verification.claim_utility import': 'from spectrue_core.verification.claims.claim_utility import',
    'from spectrue_core.verification.claim_dedup import': 'from spectrue_core.verification.claims.claim_dedup import',
    'from spectrue_core.verification.claim_frame_builder import': 'from spectrue_core.verification.claims.claim_frame_builder import',
    
    # Evidence subdirectory  
    'from spectrue_core.verification.evidence import': 'from spectrue_core.verification.evidence.evidence import',
    'from spectrue_core.verification.evidence_pack import': 'from spectrue_core.verification.evidence.evidence_pack import',
    'from spectrue_core.verification.evidence_stats import': 'from spectrue_core.verification.evidence.evidence_stats import',
    'from spectrue_core.verification.evidence_scoring import': 'from spectrue_core.verification.evidence.evidence_scoring import',
    'from spectrue_core.verification.evidence_explainability import': 'from spectrue_core.verification.evidence.evidence_explainability import',
    'from spectrue_core.verification.evidence_stance import': 'from spectrue_core.verification.evidence.evidence_stance import',
    
    # Scoring subdirectory
    'from spectrue_core.verification.rgba_aggregation import': 'from spectrue_core.verification.scoring.rgba_aggregation import',
    'from spectrue_core.verification.scoring_aggregation import': 'from spectrue_core.verification.scoring.scoring_aggregation import',
    'from spectrue_core.verification.stance_posterior import': 'from spectrue_core.verification.scoring.stance_posterior import',
    
    # Temporal subdirectory
    'from spectrue_core.verification.temporal import': 'from spectrue_core.verification.temporal.temporal import',
    
    # Targeting subdirectory
    'from spectrue_core.verification.target_selection import': 'from spectrue_core.verification.targeting.target_selection import',
}

def fix_imports_in_file(filepath):
    """Fix imports in a single file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        changes_made = []
        
        for old_import, new_import in IMPORT_MAPPINGS.items():
            if old_import in content:
                content = content.replace(old_import, new_import)
                changes_made.append((old_import, new_import))
        
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✓ Fixed {filepath}")
            for old, new in changes_made:
                print(f"  {old}")
                print(f"  → {new}")
            return True
        
        return False
    
    except Exception as e:
        print(f"✗ Error processing {filepath}: {e}")
        return False

def main():
    """Main function to fix all imports."""
    engine_root = Path(__file__).parent
    
    # Process all Python files
    python_files = list(engine_root.rglob('*.py'))
    
    fixed_count = 0
    for filepath in python_files:
        # Skip this script and __pycache__
        if '__pycache__' in str(filepath) or filepath.name == 'fix_imports.py':
            continue
        
        if fix_imports_in_file(filepath):
            fixed_count += 1
    
    print(f"\n✓ Fixed imports in {fixed_count} files")

if __name__ == '__main__':
    main()
