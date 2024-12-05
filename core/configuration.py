from torchvision import transforms
from collections.abc import Collection

class MaskConfig():

    def __init__(self, scale=100, octaves=4, persistence=0.5, lacunarity=2.0, threshold=0.0, sigma=2.0):
        self.scale = scale
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity
        self.threshold = threshold
        self.sigma = sigma

    def __iter__(self):
        # Return the instance attributes as an iterable of key-value pairs (tuples)
        for key, value in self.__dict__.items():
            if key in self.__class__.__init__.__code__.co_varnames:
                yield key, value


class ModalityConfig():

    def __init__(self, modality: str, shape: Collection[int], flip: bool, rotate: int = None):
        self.modality = modality
        self.shape = shape
        self.transforms = self.__get_transforms(flip, rotate)

    def __get_transforms(self, flip: bool, rotate: int = None):
        transformations = [transforms.CenterCrop(self.shape[1:])]
        if flip: transformations.append(transforms.RandomHorizontalFlip())
        if rotate: transformations.append(transforms.RandomRotation(rotate))
        return transforms.Compose(transformations)