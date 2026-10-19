from abc import abstractmethod
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch


class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None, interpretable_labels=None, **kwargs):
        assert len(data) == len(labels), "All input lists/tensors must be of the same length!"

        self.data = data
        self.labels = labels
        self.transform = transform
        if interpretable_labels is not None:
            self.interpretable_labels = interpretable_labels
        else:
            self.interpretable_labels = labels

        if transform is None:
            self.transform = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.data)

    @abstractmethod
    def __getitem__(self, idx):
        """Tailor to dataset"""

    @abstractmethod
    def get_original_image(self, idx):
        """Get the original image without any transformations."""


class CifarDataset(CustomDataset):
    def __init__(self, data, labels, transform=None, interpretable_labels=None):
        """
        Initialize the dataset with paths to images and corresponding labels.

        :param image_paths: List of file paths to the images.
        :param labels: List of labels corresponding to the images.
        :param transform: Any transformations to be applied to the images.
        :param interpretable_labels: Optional, human-readable labels.
        """
        super().__init__(data, labels, transform, interpretable_labels)
        self.transform = transform
        self.interpretable_labels = interpretable_labels

    def __getitem__(self, idx):
        """
        Override the __getitem__ method to load the image from disk using its path.
        
        :param idx: Index of the image and label to retrieve.
        :return: Transformed image, label, and index.
        """
        # Get the image path at the given index
        img_path = self.data[idx]

        # Load the image from disk
        img = Image.open(img_path).convert('RGB')

        # Apply transformations if provided
        if self.transform:
            img = self.transform(img)

        # Get the label for the image
        label = self.labels[idx]

        return img, label, idx

    def get_original_image(self, idx) -> Image:
        """
        Get the original image from disk without any transformations.
        
        :param idx: Index of the image to retrieve.
        :return: PIL Image object of the original image.
        """
        # Get the image path at the given index
        img_path = self.data[idx]

        # Load the image from disk
        img = Image.open(img_path).convert('RGB')

        return img


class ImagenetDataset(CustomDataset):
    def __init__(self, data, labels, transform=None, interpretable_labels=None, one_hot_encoding=False):
        super().__init__(data, labels, transform, interpretable_labels)
        num_classes = 1000  # Imagenet has 1000 classes
        if one_hot_encoding:
            labels_tensor = torch.tensor(self.labels, dtype=torch.int64)
            self.labels = torch.nn.functional.one_hot(labels_tensor, num_classes=num_classes)

    def __getitem__(self, idx):
        img = Image.open(self.data[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx] - 1

        return img, label, idx

    def get_original_image(self, idx) -> Image:
        """Get the original image without any transformations."""
        return Image.open(self.data[idx]).convert('RGB')


class TinyImagenetDataset(CustomDataset):
    def __init__(self, data, labels, transform=None, interpretable_labels=None, one_hot_encoding=True):
        super().__init__(data, labels, transform, interpretable_labels, num_classes=200)

    def __getitem__(self, idx):
        img = Image.open(self.data[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]

        return img, label, idx

    def get_original_image(self, idx) -> Image:
        return Image.open(self.data[idx]).convert('RGB')
