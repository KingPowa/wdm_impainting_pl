import numpy as np
from typing import Callable, List

class Nop(Callable):
    def __call__(self, x: np.ndarray):
        return x
    
class Compose(Callable):

    def __init__(self, tr: List[Callable] = []):
        if tr is None:
            self.tr = []
        else: self.tr = tr

    def __call__(self, x: np.ndarray):
        for t in self.tr:
            x = t(x)
        return x

import numpy as np
from scipy.ndimage import zoom

class VolumeAdapter:
    def __init__(self, target_size=(224, 224)):
        """
        Initialize the VolumeAdapter with a target width and height.
        
        Parameters:
        - target_size: Tuple (width, height) to resize the volume to.
        """
        self.target_size = target_size

    def __call__(self, volume: np.ndarray) -> np.ndarray:
        """
        Adapt the input volume to the desired dimensions and padding.
        
        Parameters:
        - volume: A 3D numpy array of shape (depth, height, width).
        
        Returns:
        - Adapted volume as a 3D numpy array.
        """
        # Ensure the input is a 3D tensor
        if len(volume.shape) < 3:
            raise ValueError("Input volume must have 3 or more dimensions: (D, H, W).")

        height, width = volume.shape[-2:]

        # Resize width and height to target dimensions
        target_height, target_width = self.target_size
        
        # Compute the scaling factors
        scale_height = target_height / height
        scale_width = target_width / width
        
        # Use the average of height and width scaling factors for depth scaling
        scale_depth = (scale_height + scale_width) / 2

        # Resize the volume
        resized_volume = zoom(volume, 
                              zoom=[1 for _ in volume.shape[:-3]] + [scale_depth, scale_height, scale_width], 
                              order=3)  # Using cubic interpolation
        
        # Get the new depth
        new_depth = resized_volume.shape[0]

        pad_depth = (4 - (new_depth % 4)) % 4
        if pad_depth > 0:
            padding = [(0,0) for _ in volume.shape[:-3]] + [(0, pad_depth), (0, 0), (0, 0)]
            adapted_volume = np.pad(resized_volume, padding, mode='constant', constant_values=0)
        else:
            adapted_volume = resized_volume
        
        return adapted_volume