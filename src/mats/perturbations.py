import re
from PIL import Image

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
    
    perturbed_sentence = sentence.lower()
    
    # Swap multi-word expressions first
    for original, replacement in spatial_swaps.items():
        if len(original.split()) > 1:  
            pattern = r'\b' + re.escape(original) + r'\b'
            perturbed_sentence = re.sub(pattern, f"__TEMP_{replacement.replace(' ', '_')}__", 
                                       perturbed_sentence, flags=re.IGNORECASE)
    
    # Then swap single words
    for original, replacement in spatial_swaps.items():
        if len(original.split()) == 1:
            pattern = r'\b' + re.escape(original) + r'\b'
            perturbed_sentence = re.sub(pattern, f"__TEMP_{replacement}__", 
                                       perturbed_sentence, flags=re.IGNORECASE)
    
    # Replace the temporary tokens with their proper form
    perturbed_sentence = re.sub(r'__TEMP_([^_]+(?:_[^_]+)*)__', 
                               lambda m: m.group(1).replace('_', ' '), 
                               perturbed_sentence)
    
    # Attempt to restore capitalization from the original
    words_original = sentence.split()
    words_perturbed = perturbed_sentence.split()
    
    final_words = []
    for orig, pert in zip(words_original, words_perturbed):
        if orig.isupper():
            final_words.append(pert.upper())
        elif orig.istitle():
            final_words.append(pert.capitalize())
        else:
            final_words.append(pert)
    final_words += words_perturbed[len(final_words):]
    
    return ' '.join(final_words)

def rotate_image(image: Image.Image, angle: int = 10) -> Image.Image:
    """
    Rotates a PIL image by a given angle.

    Args:
        image (Image.Image): The input image.
        angle (int): The angle in degrees for rotation.
                       A positive angle means counter-clockwise rotation.

    Returns:
        Image.Image: The rotated image.
    """
    # Using expand=True ensures the entire rotated image fits in the new image size
    return image.rotate(angle, expand=True)