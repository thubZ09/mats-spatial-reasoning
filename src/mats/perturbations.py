import re

def perturb_spatial_words(sentence):
    """
    Perturbs spatial words in a sentence by swapping them with their opposites.
    
    Swaps:
    - left ↔ right
    - above ↔ below  
    - top ↔ bottom
    - in front of ↔ behind
    
    Args:
        sentence (str): Input sentence containing spatial descriptions
        
    Returns:
        str: Sentence with spatial words swapped
    """
    
    spatial_swaps = {
        'left': 'right',
        'right': 'left',
        'above': 'below',
        'below': 'above',
        'top': 'bottom',
        'bottom': 'top',
        'in front of': 'behind',
        'behind': 'in front of'
    }
    
    # a copy of the sentence to work with
    perturbed_sentence = sentence.lower()
    
    for original, replacement in spatial_swaps.items():
        if len(original.split()) > 1:  
            
            pattern = r'\b' + re.escape(original) + r'\b'
            perturbed_sentence = re.sub(pattern, f"__TEMP_{replacement.replace(' ', '_')}__", 
                                     perturbed_sentence, flags=re.IGNORECASE)
    
    # single words
    for original, replacement in spatial_swaps.items():
        if len(original.split()) == 1:  # Single words
            pattern = r'\b' + re.escape(original) + r'\b'
            perturbed_sentence = re.sub(pattern, f"__TEMP_{replacement}__", 
                                     perturbed_sentence, flags=re.IGNORECASE)
    
 
    perturbed_sentence = re.sub(r'__TEMP_([^_]+(?:_[^_]+)*)__', 
                               lambda m: m.group(1).replace('_', ' '), 
                               perturbed_sentence)
    
    words_original = sentence.split()
    words_perturbed = perturbed_sentence.split()
    
    final_words = []
    for i, (orig, pert) in enumerate(zip(words_original, words_perturbed)):
        if orig.isupper():
            final_words.append(pert.upper())
        elif orig.istitle():
            final_words.append(pert.capitalize())
        else:
            final_words.append(pert)
    
    return ' '.join(final_words)