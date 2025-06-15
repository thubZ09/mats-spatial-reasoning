def check_logical_consistency(prediction_original: str, prediction_perturbed: str, caption_original: str) -> str:
    """
    Checks if two model predictions are logically consistent based on spatial opposition.

    Args:
        prediction_original: The model's output for the original caption (e.g., 'True').
        prediction_perturbed: The model's output for the perturbed caption (e.g., 'True').
        caption_original: The original caption, used to understand the context.

    Returns:
        'CONSISTENT' or 'INCONSISTENT'.
    """
    # Normalize predictions to handle variations like "True." vs "true"
    pred1 = prediction_original.lower().strip().replace('.', '')
    pred2 = prediction_perturbed.lower().strip().replace('.', '')

    # Define simple boolean keywords
    true_words = ['true', 'yes']
    false_words = ['false', 'no']

    # Determine if predictions are opposites
    is_opposite = (pred1 in true_words and pred2 in false_words) or \
                  (pred1 in false_words and pred2 in true_words)

    # For now, we assume all textual perturbations are oppositional.
    # A more advanced version could check which words were swapped.
    if is_opposite:
        return 'CONSISTENT'
    else:
        # If the predictions are the same (e.g., True/True) or nonsensical,
        # it's a failure of logical consistency.
        return 'INCONSISTENT'