from datasets import load_dataset

def get_sample_data():
    """Returns the small, hard-coded sample dataset for quick testing."""
    vsr_sample_data = [
        {'image': 'image_001.jpg', 'caption': 'The red ball is to the left of the blue box.', 'label': 1},
        {'image': 'image_002.jpg', 'caption': 'The cat is sitting above the mat on the floor.', 'label': 1},
        {'image': 'image_003.jpg', 'caption': 'The car is parked behind the large tree.', 'label': 1},
        {'image': 'image_004.jpg', 'caption': 'The bird is perched at the top of the building.', 'label': 1},
        {'image': 'image_005.jpg', 'caption': 'The book is placed below the computer monitor.', 'label': 1},
        {'image': 'image_006.jpg', 'caption': 'The dog is standing in front of the house door.', 'label': 1},
        {'image': 'image_007.jpg', 'caption': 'The flower pot is positioned to the right of the window.', 'label': 1},
        {'image': 'image_008.jpg', 'caption': 'The lamp is at the bottom of the staircase.', 'label': 1},
        {'image': 'image_009.jpg', 'caption': 'The picture frame is above the fireplace and to the left of the clock.', 'label': 1},
        {'image': 'image_010.jpg', 'caption': 'The coffee cup is on the table, in front of the laptop.', 'label': 1}
    ]
    return {'train': vsr_sample_data, 'test': vsr_sample_data[:5]}

def get_vsr_dataset():
    """Downloads and returns the real VSR dataset from Hugging Face."""
    try:
        dataset = load_dataset("juletxara/visual-spatial-reasoning")
        return dataset['random']
    except Exception as e:
        print(f"Failed to load dataset from Hugging Face: {e}")
        return None

def get_data(use_sample: bool = True):
    """
    Main data loading function.
    
    Args:
        use_sample (bool): If True, returns the small hard-coded sample dataset.
                           If False, attempts to download and return the full VSR dataset.
    
    Returns:
        A dataset dictionary.
    """
    if use_sample:
        print("Loading sample dataset.")
        return get_sample_data()
    else:
        print("Loading full VSR dataset from Hugging Face.")
        return get_vsr_dataset()