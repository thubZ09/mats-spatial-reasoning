def normalize_response(response: str) -> bool | None:
    """Convert model output to a structured boolean or None."""
    if response is None:
        return None
    
    response = response.lower().strip()
    
    # Check for affirmative answers
    if response.startswith("true") or response.startswith("yes"):
        return True
    # Check for negative answers
    elif response.startswith("false") or response.startswith("no"):
        return False
    # If the response is not clearly True or False, it's ambiguous
    else:
        return None


def check_logical_consistency(original_pred: bool | None, perturbed_pred: bool | None) -> str:
    """
    Checks if two normalized boolean predictions are logically consistent.
    For oppositional pairs (e.g., left/right), consistency means the predictions are opposites.
    """
    # If either prediction could not be parsed, we can't determine consistency.
    if original_pred is None or perturbed_pred is None:
        return 'INDETERMINATE' # Cannot determine consistency

    # For spatial opposition, consistency requires the booleans to be different.
    # e.g., original=True and perturbed=False IS CONSISTENT.
    if original_pred != perturbed_pred:
        return 'CONSISTENT'
    else:
        # e.g., original=True and perturbed=True IS INCONSISTENT.
        return 'INCONSISTENT'