"""
Custom Pytorch dataset loader classes.
class: SelectedImageFolder modifies the ImageFolder class and takes a list of the class folders to be read from the root
class: ImageList forms a dataset from a list of  path names
Author: Nic C
"""

import os
import torchvision
from torchvision.datasets.folder import IMG_EXTENSIONS
from torchvision.datasets.folder import default_loader, has_file_allowed_extension, make_dataset
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union


class SelectedImageFolder(torchvision.datasets.DatasetFolder):
    """ Custom class based on Pytorch dataset ImageFolder
        See Torch documentation for main details
        This class adds and extra argument - chosen classes.
        The function find_classes has been modified to accommodate this
        :param chosen_classes: a list of directories in the root folder
        that hold the images for a class"""

    def __init__(
            self,
            root: str,
            chosen_classes: list,
            extensions: Optional[Tuple[str, ...]] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.extensions = IMG_EXTENSIONS if is_valid_file is None else None

        classes, class_to_idx = self.find_classes(self.root, chosen_classes)
        samples = self.make_dataset(self.root, class_to_idx, self.extensions, is_valid_file)

        self.loader = loader
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    @staticmethod
    def make_dataset(
            directory: str,
            class_to_idx: Dict[str, int],
            extensions: Optional[Tuple[str, ...]] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        """Generates a list of samples of a form (path_to_sample, class).
            Overwrites parent function - See :class:`DatasetFolder` for details"""

        if class_to_idx is None:
            raise ValueError("The class_to_idx parameter cannot be None.")
        return make_dataset(directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file)

    def find_classes(self, directory: str, chosen_classes: list) -> Tuple[List[str], Dict[str, int]]:
        """Find the class folders in a dataset folder structure
        Overwrites parent function - See :class:`DatasetFolder` for details"""
        return find_classes(directory, chosen_classes)


def find_classes(directory: str, chosen_classes: list):
    """Finds the class folders in a dataset. This is an over load function to
    to allow for customisation
    Tuple[List[str], Dict[str, int]
  See :class:`DatasetFolder` for details.
  """
    dir_list = [entry.name for entry in os.scandir(directory) if entry.is_dir()]
    classes = [(name) for chosen in chosen_classes for name in dir_list if name == chosen]
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


"""Uncomment and modify the below make_data function to develop an overload function for the classes """

# def make_dataset(directory: str,
#                  class_to_idx: Optional[Dict[str, int]] = None,
#                  extensions: Optional[Union[str, Tuple[str, ...]]] = None,
#                  is_valid_file: Optional[Callable[[str], bool]] = None,
#                  ) -> List[Tuple[str, int]]:
#     """Generates a list of samples of a form (path_to_sample, class).
#     See :class:`DatasetFolder` for details.
#     Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
#     by default.
#     """
#     directory = os.path.expanduser(directory)
#
#     if class_to_idx is None:
#         _, class_to_idx = find_classes(directory)
#     elif not class_to_idx:
#         raise ValueError("'class_to_index' must have at least one entry to collect any samples.")
#
#     both_none = extensions is None and is_valid_file is None
#     both_something = extensions is not None and is_valid_file is not None
#     if both_none or both_something:
#         raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
#
#     if extensions is not None:
#         def is_valid_file(x: str) -> bool:
#             return has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]
#
#     is_valid_file = cast(Callable[[str], bool], is_valid_file)
#
#     instances = []
#     available_classes = set()
#     for target_class in sorted(class_to_idx.keys()):
#         class_index = class_to_idx[target_class]
#         target_dir = os.path.join(directory, target_class)
#         if not os.path.isdir(target_dir):
#             continue
#         for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
#             for fname in sorted(fnames):
#                 path = os.path.join(root, fname)
#                 if is_valid_file(path):
#                     item = path, class_index
#                     instances.append(item)
#
#                     if target_class not in available_classes:
#                         available_classes.add(target_class)
#
#     empty_classes = set(class_to_idx.keys()) - available_classes
#     if empty_classes:
#         msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
#         if extensions is not None:
#             msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
#         raise FileNotFoundError(msg)
#
#     return instances
