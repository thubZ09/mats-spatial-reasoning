# src/mats/metrics.py
import re
import logging
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)

def normalize_response(response):
    """
    Robustly converts model responses to boolean values with comprehensive pattern matching.
    Handles both text responses and numerical scores.
    """
    # Handle empty responses
    if response is None:
        return None
        
    # Handle numerical responses (CLIP similarity scores)
    if isinstance(response, (int, float)):
        return response > 0.5
        
    try:
        # Try converting to float (for CLIP scores)
        score = float(response)
        return score > 0.5
    except (ValueError, TypeError):
        # Not a numerical response, continue processing
        pass
    
    # Ensure we're working with a string
    if not isinstance(response, str):
        response = str(response)
        
    # Clean and standardize the response
    response = response.strip().lower()
    
    # Handle negations first (critical for accurate interpretation)
    negation_pattern = r'\b(not|isn\'?t|aren\'?t|doesn\'?t|is not|are not|does not)\b'
    if re.search(negation_pattern, response):
        if re.search(r'\b(true|yes|correct|right)\b', response):
            return False  # "not true" → False
        if re.search(r'\b(false|no|incorrect|wrong)\b', response):
            return True   # "not false" → True
    
    # Comprehensive true patterns
    true_patterns = [
        r'\b(true|yes|correct|right)\b',  # Basic affirmations
        r'\b(yep|yeah|yup|absolutely|definitely|certainly|sure)\b',  # Informal affirmations
        r'\b(accurate|valid|confirmed|affirmative|positive|agreed)\b',  # Formal affirmations
        r'^y(es)?$',  # Simple yes
        r'is (correct|right|true|accurate)',  # Statement forms
        r'^true$',  # Exact match
        r'that\'?s (correct|right|true)',  # Common phrasing
        r'\b(correct answer|right one)\b'  # Specific phrases
    ]
    
    # Comprehensive false patterns
    false_patterns = [
        r'\b(false|no|incorrect|wrong)\b',  # Basic negatives
        r'\b(nope|nah|negative|absolutely not|definitely not|certainly not)\b',  # Informal negatives
        r'\b(inaccurate|invalid|disconfirmed|negative|disagreed)\b',  # Formal negatives
        r'^n(o)?$',  # Simple no
        r'is (incorrect|wrong|false|inaccurate)',  # Statement forms
        r'^false$',  # Exact match
        r'that\'?s (incorrect|wrong|false)',  # Common phrasing
        r'\b(incorrect answer|wrong one)\b'  # Specific phrases
    ]
    
    # Check for true patterns
    for pattern in true_patterns:
        if re.search(pattern, response):
            return True
    
    # Check for false patterns
    for pattern in false_patterns:
        if re.search(pattern, response):
            return False
    
    # Special handling for spatial terms to avoid confusion
    spatial_terms = ['left', 'right', 'above', 'below', 'front', 'behind', 'beside']
    if any(term in response for term in spatial_terms):
        logger.debug(f"Spatial term in response: '{response}'")
        return None
    
    # Log unhandled responses for improvement
    logger.warning(f"Unhandled response pattern: '{response}'")
    return None

def check_text_consistency(orig_norm, pert_norm):
    """
    For text perturbation audits, consistency means the answers should be OPPOSITE.
    Text swaps change the meaning (left→right), so correct responses should flip.
    """
    if orig_norm is None or pert_norm is None:
        return 'INDETERMINATE'
    return 'CONSISTENT' if orig_norm != pert_norm else 'INCONSISTENT'

def check_visual_consistency(orig_norm, pert_norm):
    """
    For visual perturbation audits, consistency means answers should be THE SAME.
    Minor rotations shouldn't change spatial relationships.
    """
    if orig_norm is None or pert_norm is None:
        return 'INDETERMINATE'
    return 'CONSISTENT' if orig_norm == pert_norm else 'INCONSISTENT'
    
def check_coordinated_consistency(orig_norm, pert_norm):
    """
    For coordinated perturbation audits, consistency means answers should be THE SAME.
    Flipping both image and text should preserve truth value.
    """
    if orig_norm is None or pert_norm is None:
        return 'INDETERMINATE'
    return 'CONSISTENT' if orig_norm == pert_norm else 'INCONSISTENT'

def generate_summary_table(results):
    """Generate comprehensive statistics from audit results"""
    if not results:
        logger.warning("Empty results passed to generate_summary_table")
        return {}
    
    summary = {
        'total_samples': len(results),
        'consistent': 0,
        'inconsistent': 0,
        'indeterminate': 0,
        'consistency_rate': 0.0,
        'failure_modes': defaultdict(int),
        'relation_breakdown': defaultdict(lambda: {
            'total': 0,
            'consistent': 0,
            'rate': 0.0
        })
    }
    
    # Process each result
    for r in results:
        consistency = r.get('consistency', 'INDETERMINATE')
        
        # Count consistency types
        if consistency == 'CONSISTENT':
            summary['consistent'] += 1
        elif consistency == 'INCONSISTENT':
            summary['inconsistent'] += 1
            # Track failure modes
            rel_type = r.get('relation_type', 'unknown')
            summary['failure_modes'][rel_type] += 1
        else:
            summary['indeterminate'] += 1
        
        # Track per-relation performance
        rel_type = r.get('relation_type', 'unknown')
        rel_data = summary['relation_breakdown'][rel_type]
        rel_data['total'] += 1
        if consistency == 'CONSISTENT':
            rel_data['consistent'] += 1
    
    # Calculate rates
    valid_samples = summary['total_samples'] - summary['indeterminate']
    if valid_samples > 0:
        summary['consistency_rate'] = summary['consistent'] / valid_samples
    
    # Calculate per-relation rates
    for rel, data in summary['relation_breakdown'].items():
        if data['total'] > 0:
            data['rate'] = data['consistent'] / data['total']
    
    return summary

def test_normalization():
    """Comprehensive test suite for response normalization"""
    test_cases = [
        # True responses
        ("true", True),
        ("yes", True),
        ("correct", True),
        ("that's right", True),
        ("absolutely", True),
        ("definitely true", True),
        ("it is correct", True),
        ("affirmative", True),
        ("positive", True),
        ("y", True),
        ("true answer", True),
        
        # False responses
        ("false", False),
        ("no", False),
        ("incorrect", False),
        ("that's wrong", False),
        ("absolutely not", False),
        ("definitely false", False),
        ("it is incorrect", False),
        ("negative", False),
        ("n", False),
        ("wrong answer", False),
        
        # Negations
        ("not true", False),
        ("isn't correct", False),
        ("not right", False),
        ("not false", True),
        ("isn't wrong", True),
        ("doesn't incorrect", True),
        
        # Numerical scores
        (0.6, True),
        (0.4, False),
        ("0.7", True),
        ("0.3", False),
        
        # Spatial terms (should be indeterminate)
        ("left of", None),
        ("on the right", None),
        ("above and below", None),
        
        # Unhandled cases
        ("maybe", None),
        ("I don't know", None),
        ("can't tell", None),
    ]
    
    passed = 0
    for response, expected in test_cases:
        result = normalize_response(response)
        if result == expected:
            passed += 1
        else:
            print(f"FAIL: '{response}' -> {result} (expected {expected})")
    
    print(f"\nNormalization Test Results: {passed}/{len(test_cases)} passed")
    return passed == len(test_cases)

# Run tests when module is executed directly
if __name__ == "__main__":
    print("Running metrics tests...")
    test_normalization()