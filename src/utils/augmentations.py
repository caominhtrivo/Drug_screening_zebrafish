# src/utils/augmentations.py
import numpy as np

def apply_augmentations(x):
    """
    Applies a suite of temporal data augmentations to a feature sequence.
    
    Args:
        x (np.ndarray): The input feature sequence of shape (seq_len, feature_dim).
    Returns:
        np.ndarray: The augmented sequence.
    """
    # 1. Jittering: Adds small Gaussian noise to simulate sensor/tracking variance
    noise = np.random.normal(loc=0, scale=0.05, size=x.shape)
    x = x + noise

    # 2. Scaling: Multiplies by a random factor to simulate distance variations
    scale = np.random.normal(loc=1.0, scale=0.1)
    x = x * scale

    # 3. Time Masking: Randomly zeros out a block of frames (simulates occlusion)
    if np.random.rand() > 0.5:
        mask_len = np.random.randint(5, 15) 
        if x.shape[0] > mask_len:
            mask_start = np.random.randint(0, x.shape[0] - mask_len)
            x[mask_start:mask_start+mask_len, :] = 0

    return x