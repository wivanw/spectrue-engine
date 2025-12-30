from typing import List
from spectrue_core.schema.evidence import EvidenceItem, EvidenceStance
from spectrue_core.schema.scoring import ConsensusState

def calculate_consensus(evidence_list: List[EvidenceItem]) -> ConsensusState:
    """
    Calculates the Scientific Consensus latent variable based on evidence.
    
    Consensus is modeled by:
    1. Cross-source agreement (Support vs Refute ratio)
    2. Source independence (Unique domains)
    3. Temporal stability (placeholder)
    
    Returns:
        ConsensusState with score [0,1], stability, and source count.
        Score 1.0 = Strong Consensus Support.
        Score 0.0 = Strong Consensus Refutation.
        Score 0.5 = No Consensus / Split.
    """
    if not evidence_list:
        return ConsensusState(score=0.5, stability=0.0, source_count=0)

    # 1. Source Independence
    domains = set(e.domain for e in evidence_list if e.domain)
    source_count = len(domains)
    
    # 2. Agreement
    supports = 0
    refutes = 0
    valid_evidence_count = 0
    
    for e in evidence_list:
        if e.stance == EvidenceStance.SUPPORT:
            supports += 1
            valid_evidence_count += 1
        elif e.stance == EvidenceStance.REFUTE:
            refutes += 1
            valid_evidence_count += 1
            
    if valid_evidence_count == 0:
        agreement_score = 0.5 # Neutral / No consensus signal
    else:
        # Simple ratio: Supports / Total
        # 1.0 = Unanimous Support
        # 0.0 = Unanimous Refutation
        # 0.5 = Split
        agreement_score = supports / valid_evidence_count

    # 3. Stability
    # For now, we assume standard stability. 
    # Temporal logic would refine this based on timeliness consistency.
    stability = 1.0 

    return ConsensusState(
        score=agreement_score, 
        stability=stability, 
        source_count=source_count
    )
