# src/mats/metrics.py
import re

def normalize_response(response: str):
    """Convert free-text responses to a structured boolean or None."""
    if response is None or not isinstance(response, str):
        return None
    response = response.lower().strip()
    if re.search(r'\b(true|yes|correct|right)\b', response):
        return True
    if re.search(r'\b(false|no|incorrect|wrong)\b', response):
        return False
    return None

def check_text_consistency(orig_norm, pert_norm):
    """For text swaps, consistency means the answers should be OPPOSITE."""
    if orig_norm is None or pert_norm is None:
        return 'INDETERMINATE'
    return 'CONSISTENT' if orig_norm != pert_norm else 'INCONSISTENT'

def check_visual_consistency(orig_norm, pert_norm):
    """For visual rotation, consistency means the answers should be THE SAME."""
    if orig_norm is None or pert_norm is None:
        return 'INDETERMINATE'
    return 'CONSISTENT' if orig_norm == pert_norm else 'INCONSISTENT'
    
def check_coordinated_consistency(orig_norm, pert_norm):
    """For coordinated flip+swap, consistency means answers should be THE SAME."""
    if orig_norm is None or pert_norm is None:
        return 'INDETERMINATE'
    return 'CONSISTENT' if orig_norm == pert_norm else 'INCONSISTENT'

def generate_summary_table(results):
    """Generate summary statistics from audit results."""
    if not results: return {}
    total = len(results)
    consistent = sum(1 for r in results if r.get('consistency') == 'CONSISTENT')
    inconsistent = sum(1 for r in results if r.get('consistency') == 'INCONSISTENT')
    indeterminate = sum(1 for r in results if r.get('consistency') == 'INDETERMINATE')
    failure_modes = {}
    for r in results:
        if r.get('consistency') == 'INCONSISTENT':
            rel_type = r.get('relation_type', 'unknown')
            failure_modes[rel_type] = failure_modes.get(rel_type, 0) + 1
    return {
        'total_samples': total,
        'consistent': consistent,
        'inconsistent': inconsistent,
        'indeterminate': indeterminate,
        'consistency_rate': consistent / (total - indeterminate) if (total - indeterminate) > 0 else 0,
        'failure_modes': failure_modes
    }