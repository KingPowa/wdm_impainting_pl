from torchvision import transforms
from collections.abc import Collection

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