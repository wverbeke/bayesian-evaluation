"""Data loader for Mapillary.

The loader will load images containing individual traffic sign patches. This assumes that the Mapillary
data set has already been preprocessed to extract all traffic signs.
"""
import os
import json
import math
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from mapillary_data_loader.preproc_mapillary import TRAIN_ANNOTATION_LIST_PATH, EVAL_ANNOTATION_LIST_PATH, read_annotation
from mapillary_data_loader.make_class_list import mapillary_class_list

class MapillaryDataset(Dataset):

    # kwargs are ignored but make it callable in the same way as standard pytorch loaders
    def __init__(self, transform, train, **kwargs):
        if train:
            annotation_dict = read_annotation(TRAIN_ANNOTATION_LIST_PATH)
        else:
            annotation_dict = read_annotation(EVAL_ANNOTATION_LIST_PATH)
        self._image_paths = []
        self._annotations = []
        class_list = mapillary_class_list()
        for image_path, class_name in annotation_dict.items():
            class_name = class_name[0]
            self._image_paths.append(image_path)
            self._annotations.append(class_list.index(class_name))
        self._transform = transform

    def __len__(self):
        return len(self._image_paths)

    def __getitem__(self, index):

        # Read and transform the image.
        image = Image.open(self._image_paths[index]).convert("RGB")
        image_tensor = self._transform(image)

        return image_tensor, self._annotations[index]
