import os
from torchvision import transforms
import torch
import sys
from typing import List, Dict, Tuple, Any
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


def class_finder(directory: Path):
    """
    Finds and maps class names to indices based on subdirectories in the given directory.

    This function scans the provided directory for subdirectories, treats each subdirectory
    as a class, and returns a list of class names and a dictionary mapping class names to
    their corresponding indices.

    Args:
        directory (Path): Path to the directory containing subdirectories representing classes.

    Returns:
        Tuple[List[str], Dict[str, int]]: A tuple containing:
            - A list of class names (sorted alphabetically).
            - A dictionary mapping class names to their corresponding indices.

    Raises:
        FileNotFoundError: If the directory does not contain any subdirectories (no classes found).

    Example:
        >>> directory = Path("path/to/dataset")
        >>> classes, class_to_idx = class_finder(directory)
        >>> print(classes)
        ['cat', 'dog']
        >>> print(class_to_idx)
        {'cat': 0, 'dog': 1}
    """

    # Get sorted list of class names from subdirectories
    classes: List[str] = sorted(i.name for i in os.scandir(directory) if i.is_dir())

    # Raise an error if no classes are found
    if not classes:
        raise FileNotFoundError(
            f"This directory does not have any classes: {directory}"
        )

    # Create a dictionary mapping class names to indices
    class_to_idx: Dict[str, int] = {name: idx for idx, name in enumerate(classes)}

    return classes, class_to_idx


class ImageFolderCustom(Dataset):
    """
    A custom dataset class for loading images from a directory structure.

    This class extends `torch.utils.data.Dataset` and is designed to load images
    from a directory where images are organized into subdirectories by class.
    It supports applying transformations to the images and provides methods for
    accessing images and their corresponding class labels.

    Attributes:
        paths (list of Path): List of paths to all image files in the target directory.
        transform (callable, optional): A function/transform to apply to the images.
        classes (list): List of class names derived from subdirectory names.
        class_to_idx (dict): A dictionary mapping class names to indices.

    Methods:
        __init__(target_dir, transform):
            Initializes the dataset with the target directory and transformations.
        load_image(index):
            Loads and returns the image at the specified index.
        __len__():
            Returns the total number of images in the dataset.
        __getitem__(index):
            Returns the image and its corresponding class index at the specified index.

    Example:
        >>> from torchvision import transforms
        >>> transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        >>> dataset = ImageFolderCustom(target_dir="path/to/images", transform=transform)
        >>> len(dataset)
        1000
        >>> image, label = dataset[0]
        >>> image.shape
        torch.Size([3, 224, 224])
        >>> label
        0
    """

    train_transforms = transforms.Compose(
        [
            transforms.Resize(size=(150, 150)),
            transforms.ColorJitter(0.4, 0.5, 0.5, 0.2),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(size=(150, 150)),
            transforms.ToTensor(),
            transforms.Normalize((0.425, 0.415, 0.405), (0.205, 0.205, 0.205)),
        ]
    )

    test_transforms = transforms.Compose(
        [
            transforms.Resize((150, 150)),
            transforms.ToTensor(),
            transforms.Normalize((0.425, 0.415, 0.405), (0.255, 0.245, 0.235)),
        ]
    )

    def __init__(self, target_dir: str, train_transform: bool):
        """
        Initializes the ImageFolderCustom dataset.

        Args:
            target_dir: Path to the directory containing images organized by class.
            transform: A function/transform to apply to the images (depends on train/val).
        """

        self.paths = list(Path(target_dir).glob("*/*.jpg"))

        if train_transform:
            self.transform = self.train_transforms
        else:
            self.transform = self.test_transforms

        self.classes, self.classes_to_idx = class_finder(target_dir)

    def load_image(self, index: int) -> Image.Image:
        """
        Loads and returns the image at the specified index.

        Args:
            index (int): Index of the image to load.

        Returns:
            Image.Image: The loaded image.
        """
        image_path = self.paths[index]

        return Image.open(image_path)

    def __len__(self) -> int:
        """
        Returns the total number of images in the dataset.

        Returns:
            int: Number of images in the dataset.
        """

        return len(self.paths)

    def __getitem__(self, indx: int) -> Tuple[Image.Image, int]:
        """
        Returns the image and its corresponding class index at the specified index.

        Args:
            index (int): Index of the image to retrieve.

        Returns:
            Tuple[Image.Image, int]: A tuple containing the image (transformed if applicable)
                                     and the class index.
        """
        img = self.load_image(indx)

        class_name = self.paths[indx].parent.name

        class_idx = self.classes_to_idx[class_name]

        if self.transform:
            return self.transform(img), class_idx
        else:
            return img, class_idx
        
class InferCustom(Dataset):
    """
    A custom dataset class for loading images from a directory structure.

    This class extends `torch.utils.data.Dataset` and is designed to load images
    from a directory where images are organized into subdirectories by class.
    It supports applying transformations to the images and provides methods for
    accessing images and their corresponding class labels.

    Attributes:
        paths (list of Path): List of paths to all image files in the target directory.
        transform (callable, optional): A function/transform to apply to the images.
        classes (list): List of class names derived from subdirectory names.
        class_to_idx (dict): A dictionary mapping class names to indices.

    Methods:
        __init__(target_dir, transform):
            Initializes the dataset with the target directory and transformations.
        load_image(index):
            Loads and returns the image at the specified index.
        __len__():
            Returns the total number of images in the dataset.
        __getitem__(index):
            Returns the image and its corresponding class index at the specified index.

    Example:
        >>> from torchvision import transforms
        >>> transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        >>> dataset = ImageFolderCustom(target_dir="path/to/images", transform=transform)
        >>> len(dataset)
        1000
        >>> image, label = dataset[0]
        >>> image.shape
        torch.Size([3, 224, 224])
        >>> label
        0
    """

    def __init__(self, target_dir: str):
        """
        Initializes the ImageFolderCustom dataset.

        Args:
            target_dir: Path to the directory containing images.
            transform: A function/transform to apply to the images.
        """

        self.path = list(Path(target_dir).glob("*"))

        self.transform = transforms.Compose(
        [
            transforms.Resize((150, 150)),
            transforms.ToTensor(),
            transforms.Normalize((0.425, 0.415, 0.405), (0.255, 0.245, 0.235)),
        ]
    )

    def load_image(self, index: int) -> Image.Image:
        """
        Loads and returns the image at the specified index.

        Args:
            index (int): Index of the image to load.

        Returns:
            Image.Image: The loaded image.
        """
        image_path = self.path[index]

        return Image.open(image_path)

    def __len__(self) -> int:
        """
        Returns the total number of images in the dataset.

        Returns:
            int: Number of images in the dataset.
        """

        return len(self.path)

    def __getitem__(self, indx: int) -> Tuple[Image.Image, int]:
        """
        Returns the image and its corresponding class index at the specified index.

        Args:
            index (int): Index of the image to retrieve.

        Returns:
            Tuple[Image.Image, int]: A tuple containing the image (transformed if applicable)
                                     and the class index.
        """
        img = self.load_image(indx)

        return self.transform(img)

def init_dataloader(
    dataset: Any, batch_size: int, shuffle: bool = True, num_workers: int = 6
):
    """Initialize torch dataloader from dataset

    Args:
        dataset (Any): dataset for dataloader
        batch_size (int): -
        shuffle (bool, optional): flag for shuffling data. Defaults to True.
        num_workers (int, optional): Defaults to 6.

    Returns:
        torch.utils.data.Dataloader: usable torch dataloader
    """
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

def main():
    print(torch.__name__)

if __name__ == "__main__":
    print(sys.modules)
