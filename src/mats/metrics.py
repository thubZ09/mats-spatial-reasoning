import re
import logging
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ModelResponse(Enum):
    """enum for parsed model responses"""
    TRUE = "TRUE"
    FALSE = "FALSE"
    UNKNOWN = "UNKNOWN"

@dataclass
class EvaluationResult:
    """container for complete evaluation results"""
    model_name: str
    scs: float
    iar: float
    coverage: float
    macs: float
    scs_ci: Tuple[float, float]
    iar_ci: Tuple[float, float]
    per_relation_scs: Dict[str, float]
    per_category_iar: Dict[str, float]
    raw_responses: List[Dict]
    # Legacy compatibility
    consistency_summary: Dict = None

def normalize_response(response):
    """
    converts model responses to boolean values with pattern matching
    handles text responses and numerical scores
    
    args:
        response: model output (str, float, int) 
    returns:
        true/false/none (None = UNKNOWN)
    """
    #handle empty responses
    if response is None:
        return None
        
    #handle numerical responses=
    if isinstance(response, (int, float)):
        return response > 0.5
        
    try:
        #try converting to float
        score = float(response)
        return score > 0.5
    except (ValueError, TypeError):

        pass

    if not isinstance(response, str):
        response = str(response)
        
    #clean and standardize the response
    response = response.strip().lower()
    
    #handle negations first (for accurate interpretation)
    negation_pattern = r'\b(not|isn\'?t|aren\'?t|doesn\'?t|is not|are not|does not)\b'
    if re.search(negation_pattern, response):
        if re.search(r'\b(true|yes|correct|right)\b', response):
            return False  #"not true" → False
        if re.search(r'\b(false|no|incorrect|wrong)\b', response):
            return True   #"not false" → True
    
    #comprehensive true patterns
    true_patterns = [
        r'\b(true|yes|correct|right)\b',  #basic affirmations
        r'\b(yep|yeah|yup|absolutely|definitely|certainly|sure)\b',  #informal
        r'\b(accurate|valid|confirmed|affirmative|positive|agreed)\b',  #formal
        r'^y(es)?$',  #simple yes
        r'is (correct|right|true|accurate)',  #statement forms
        r'^true$',  #exact match
        r'that\'?s (correct|right|true)',  #common phrasing
        r'\b(correct answer|right one)\b'  #specific phrases
    ]
    
    #comprehensive false patterns
    false_patterns = [
        r'\b(false|no|incorrect|wrong)\b',  #basic negatives
        r'\b(nope|nah|negative|absolutely not|definitely not|certainly not)\b',  #informal
        r'\b(inaccurate|invalid|disconfirmed|negative|disagreed)\b',  #formal
        r'^n(o)?$',  #simple no
        r'is (incorrect|wrong|false|inaccurate)',  #statement forms
        r'^false$',  #exact match
        r'that\'?s (incorrect|wrong|false)',  #common phrasing
        r'\b(incorrect answer|wrong one)\b'  #specific phrases
    ]
    
    #check for true patterns
    for pattern in true_patterns:
        if re.search(pattern, response):
            return True
    
    #check for false patterns
    for pattern in false_patterns:
        if re.search(pattern, response):
            return False
    
    #special handling for spatial terms to avoid confusion
    spatial_terms = ['left', 'right', 'above', 'below', 'front', 'behind', 'beside']
    if any(term in response for term in spatial_terms):
        logger.debug(f"Spatial term in response: '{response}'")
        return None
    
    #log unhandled responses for improvement
    logger.warning(f"Unhandled response pattern: '{response}'")
    return None

def check_text_consistency(orig_norm, pert_norm):
    """
    for text perturbation audits, consistency means the answers should be OPPOSITE
    text swaps change the meaning (left→right), so correct responses should flip
    """
    if orig_norm is None or pert_norm is None:
        return 'INDETERMINATE'
    return 'CONSISTENT' if orig_norm != pert_norm else 'INCONSISTENT'


def check_visual_consistency(orig_norm, pert_norm):
    """
    for visual perturbation audits, consistency means answers should be THE SAME
    minor rotations shouldn't change spatial relationships
    """
    if orig_norm is None or pert_norm is None:
        return 'INDETERMINATE'
    return 'CONSISTENT' if orig_norm == pert_norm else 'INCONSISTENT'


def check_coordinated_consistency(orig_norm, pert_norm):
    """
    for coordinated perturbation audits, consistency means answers should be THE SAME
    flipping both image and text should preserve truth value
    """
    if orig_norm is None or pert_norm is None:
        return 'INDETERMINATE'
    return 'CONSISTENT' if orig_norm == pert_norm else 'INCONSISTENT'


def generate_summary_table(results):
    """
    generate comprehensive statistics from audit results
    legacy function for compatibility with original code
    """
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
    
    #process each result
    for r in results:
        consistency = r.get('consistency', 'INDETERMINATE')
        
        #count consistency types
        if consistency == 'CONSISTENT':
            summary['consistent'] += 1
        elif consistency == 'INCONSISTENT':
            summary['inconsistent'] += 1
            #track failure modes
            rel_type = r.get('relation_type', 'unknown')
            summary['failure_modes'][rel_type] += 1
        else:
            summary['indeterminate'] += 1
        
        #track per-relation performance
        rel_type = r.get('relation_type', 'unknown')
        rel_data = summary['relation_breakdown'][rel_type]
        rel_data['total'] += 1
        if consistency == 'CONSISTENT':
            rel_data['consistent'] += 1
    
    #calculate rates
    valid_samples = summary['total_samples'] - summary['indeterminate']
    if valid_samples > 0:
        summary['consistency_rate'] = summary['consistent'] / valid_samples
    
    #calculate per-relation rates
    for rel, data in summary['relation_breakdown'].items():
        if data['total'] > 0:
            data['rate'] = data['consistent'] / data['total']
    
    return summary

class MATSMetrics:
    """
    implementation of MATS behavioral metrics
    """
    
    def __init__(self):
        """initialize MATS metrics calculator"""
        self.relation_types = ['left', 'right', 'above', 'below', 'front', 'behind']
        self.categories = ['color', 'object', 'spatial']
    
    def parse_response(self, response: str) -> ModelResponse:
        """
        parse model output to TRUE/FALSE/UNKNOWN using robust normalization
        
        args:
            response: raw model output string    
        returns:
            parsed ModelResponse enum
        """
        norm = normalize_response(response)
        
        if norm is True:
            return ModelResponse.TRUE
        elif norm is False:
            return ModelResponse.FALSE
        else:
            return ModelResponse.UNKNOWN
    
    def compute_scs(
        self,
        original_responses: List[ModelResponse],
        inverted_responses: List[ModelResponse],
        exclude_unknown: bool = True
    ) -> Tuple[float, int, int]:
        """
        compute Spatial Consistency Score (SCS)
        
        SCS = fraction of examples where response flips after predicate inversion
        SCS = 1 - (fraction where response stays same)
        
        low SCS (1-3%) indicates pathological truth bias
        high SCS (60-70%) indicates proper spatial reasoning
        
        args:
            original_responses: responses to original statements
            inverted_responses: responses to inverted statements
            exclude_unknown: whether to exclude UNKNOWN from calculation    
        returns:
            (scs_score, num_flips, total_valid)
        """
        assert len(original_responses) == len(inverted_responses)
        
        flips = 0
        total = 0
        
        for orig, inv in zip(original_responses, inverted_responses):
            if exclude_unknown and (orig == ModelResponse.UNKNOWN or 
                                   inv == ModelResponse.UNKNOWN):
                continue
            
            total += 1
            
            #count as flip if responses differ
            if orig != inv:
                flips += 1
        
        scs = flips / total if total > 0 else 0.0
        return scs, flips, total
    
    def compute_scs_by_relation(
        self,
        data: List[Dict]
    ) -> Dict[str, float]:
        """
        compute SCS broken down by relation type
        
        args:
            data: list of dicts with 'relation', 'original_response', 'inverted_response'    
        returns:
            dict mapping relation_type -> scs_score
        """
        relation_results = {rel: {'flips': 0, 'total': 0} for rel in self.relation_types}
        
        for item in data:
            relation = item.get('relation', 'unknown')
            orig = item['original_response']
            inv = item['inverted_response']
            
            if relation not in relation_results:
                continue
            
            #skip UNKNOWN
            if orig == ModelResponse.UNKNOWN or inv == ModelResponse.UNKNOWN:
                continue
            
            relation_results[relation]['total'] += 1
            if orig != inv:
                relation_results[relation]['flips'] += 1
        
        #compute SCS per relation
        scs_by_relation = {}
        for rel, stats in relation_results.items():
            if stats['total'] > 0:
                scs_by_relation[rel] = stats['flips'] / stats['total']
            else:
                scs_by_relation[rel] = 0.0
        
        return scs_by_relation
    
    def compute_iar(
        self,
        responses: List[ModelResponse],
        exclude_unknown: bool = True
    ) -> Tuple[float, int, int]:
        """
        compute Incorrect Agreement Rate (IAR)
        
        IAR = fraction of absurd statements marked TRUE
        
        high IAR (75-80%) indicates pathological truth bias
        low IAR (8-12%) indicates proper rejection behavior
        
        args:
            responses: responses to absurd statements (all should be FALSE)
            exclude_unknown: whether to exclude UNKNOWN from calculation    
        returns:
            (iar_score, num_incorrect_agreements, total_valid)
        """
        incorrect_agreements = 0
        total = 0
        
        for resp in responses:
            if exclude_unknown and resp == ModelResponse.UNKNOWN:
                continue
            
            total += 1
            if resp == ModelResponse.TRUE:
                incorrect_agreements += 1
        
        iar = incorrect_agreements / total if total > 0 else 0.0
        return iar, incorrect_agreements, total
    
    def compute_iar_by_category(
        self,
        data: List[Dict]
    ) -> Dict[str, float]:
        """
        compute IAR broken down by category (color, object, spatial)
        
        args:
            data: list of dicts with 'category' and 'response'     
        returns:
            dict mapping category -> iar_score
        """
        category_results = {cat: {'true_count': 0, 'total': 0} for cat in self.categories}
        
        for item in data:
            category = item.get('category', 'unknown')
            response = item['response']
            
            if category not in category_results:
                continue
            
            #skip UNKNOWN
            if response == ModelResponse.UNKNOWN:
                continue
            
            category_results[category]['total'] += 1
            if response == ModelResponse.TRUE:
                category_results[category]['true_count'] += 1
        
        #compute IAR per category
        iar_by_category = {}
        for cat, stats in category_results.items():
            if stats['total'] > 0:
                iar_by_category[cat] = stats['true_count'] / stats['total']
            else:
                iar_by_category[cat] = 0.0
        
        return iar_by_category
    
    def compute_coverage(
        self,
        responses: List[ModelResponse]
    ) -> float:
        """
        compute coverage: fraction of non-UNKNOWN responses
        
        args:
            responses: list of model responses   
        returns:
            coverage rate [0, 1]
        """
        if len(responses) == 0:
            return 0.0
        
        known = sum(1 for r in responses if r != ModelResponse.UNKNOWN)
        return known / len(responses)
    
    def evaluate_model(
        self,
        model_name: str,
        scs_data: List[Dict],
        iar_data: List[Dict],
        compute_ci: bool = True,
        n_bootstrap: int = 1000
    ) -> EvaluationResult:
        """
        complete evaluation of a model on MATS
        
        args:
            model_name: name of the model
            scs_data: data for SCS computation (with 'original_raw', 'inverted_raw')
            iar_data: data for IAR computation (with 'raw_response')
            compute_ci: whether to compute confidence intervals
            n_bootstrap: number of bootstrap samples for CIs    
        returns:
            EvaluationResult object with all metrics
        """
        #parse responses using robust normalization
        for item in scs_data:
            item['original_response'] = self.parse_response(item['original_raw'])
            item['inverted_response'] = self.parse_response(item['inverted_raw'])
        
        for item in iar_data:
            item['response'] = self.parse_response(item['raw_response'])
        
        #compute SCS
        orig_responses = [item['original_response'] for item in scs_data]
        inv_responses = [item['inverted_response'] for item in scs_data]
        scs, _, _ = self.compute_scs(orig_responses, inv_responses)
        
        #compute IAR
        iar_responses = [item['response'] for item in iar_data]
        iar, _, _ = self.compute_iar(iar_responses)
        
        #compute coverage
        all_responses = orig_responses + inv_responses + iar_responses
        coverage = self.compute_coverage(all_responses)
                
        per_relation_scs = self.compute_scs_by_relation(scs_data) 
        per_category_iar = self.compute_iar_by_category(iar_data)
        
        #compute confidence intervals 
        scs_ci = (0.0, 0.0)
        iar_ci = (0.0, 0.0)
        
        #generate legacy consistency summary for backward compatibility
        legacy_results = []
        for item in scs_data:
            orig_norm = normalize_response(item['original_raw'])
            inv_norm = normalize_response(item['inverted_raw'])
            consistency = check_text_consistency(orig_norm, inv_norm)
            legacy_results.append({
                'consistency': consistency,
                'relation_type': item.get('relation', 'unknown')
            })
        consistency_summary = generate_summary_table(legacy_results)
        
        return EvaluationResult(
            model_name=model_name,
            scs=scs,
            iar=iar,
            coverage=coverage,
            macs=macs,
            scs_ci=scs_ci,
            iar_ci=iar_ci,
            per_relation_scs=per_relation_scs,
            per_category_iar=per_category_iar,
            raw_responses={'scs_data': scs_data, 'iar_data': iar_data},
            consistency_summary=consistency_summary
        )
    
    def print_summary(self, result: EvaluationResult):
        """print formatted summary of evaluation results"""
        print(f"\n{'='*60}")
        print(f"MATS Evaluation Results: {result.model_name}")
        print(f"{'='*60}")
        print(f"  Spatial Consistency Score (SCS): {result.scs*100:>6.2f}%")
        print(f"  Incorrect Agreement Rate (IAR):  {result.iar*100:>6.2f}%")
        print(f"  Coverage:                         {result.coverage*100:>6.2f}%")
        
        print(f"\nPer-Relation SCS:")
        for relation, score in result.per_relation_scs.items():
            print(f"  {relation:>8}: {score*100:>6.2f}%")
        
        print(f"\nPer-Category IAR:")
        for category, score in result.per_category_iar.items():
            print(f"  {category:>8}: {score*100:>6.2f}%")
        
        #legacy summary
        if result.consistency_summary:
            print(f"\nLegacy Consistency Summary:")
            print(f"  Consistency Rate: {result.consistency_summary['consistency_rate']*100:.1f}%")
            print(f"  Note: Inverse of SCS (high consistency = low SCS)")
        
        print(f"{'='*60}\n")

def test_normalization():
    """comprehensive tests for response normalization"""
    test_cases = [
        #true responses
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
        
        #false responses
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
        
        #negations
        ("not true", False),
        ("isn't correct", False),
        ("not right", False),
        ("not false", True),
        ("isn't wrong", True),
        
        #numerical scores
        (0.6, True),
        (0.4, False),
        ("0.7", True),
        ("0.3", False),
        
        #spatial terms 
        ("left of", None),
        ("on the right", None),
        ("above and below", None),
        
        #unhandled cases
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
    
    print(f"\nNormalization test results: {passed}/{len(test_cases)} passed")
    return passed == len(test_cases)


def test_scs_computation():
    """test SCS metric computation"""
    metrics = MATSMetrics()
    
    #perfect flipping (high SCS, good behavior - Case 1) 
    print("\nPerfect flipping (contrastive encoders)")
    orig = [ModelResponse.TRUE, ModelResponse.FALSE, ModelResponse.TRUE]
    inv = [ModelResponse.FALSE, ModelResponse.TRUE, ModelResponse.FALSE]
    scs, flips, total = metrics.compute_scs(orig, inv)
    print(f"  SCS: {scs*100:.1f}% (expected ~100%)")
    assert scs == 1.0, "Perfect flipping should give SCS=1.0"
    
    #no flipping (low SCS, pathological bias - Case 2)
    print("\nNo flipping (generative VLMs with truth bias)")
    orig = [ModelResponse.TRUE, ModelResponse.TRUE, ModelResponse.TRUE]
    inv = [ModelResponse.TRUE, ModelResponse.TRUE, ModelResponse.TRUE]
    scs, flips, total = metrics.compute_scs(orig, inv)
    print(f"  SCS: {scs*100:.1f}% (expected ~0%)")
    assert scs == 0.0, "No flipping should give SCS=0.0"
    
    print("\nSCS tests passed")


if __name__ == "__main__":
    print("Running combined metrics tests...")
    print("\n" + "="*60)
    print("Testing response normalization...")
    print("="*60)
    test_normalization()
    
    print("\n" + "="*60)
    print("Testing SCS computation...")
    print("="*60)
    test_scs_computation()
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)