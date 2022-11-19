"""Some simple functionality for reading the list of Mapillary classes.

By using the functionality here we can easily get the list of Mapillary classes anywhere in the code, and
also guarantee the order stays correct and consistent.
"""
import os
from functiools import lru_cache
from mapillary_data_loader.preproc_mapillary import read_annotation, TRAIN_ANNOTATION_LIST_PATH, EVAL_ANNOTATION_LIST_PATH, BASE_PATH

MAPILLARY_CLASS_LIST_PATH = os.path.join(BASE_PATH, "mapillary_class_list.txt")

def write_mapillary_class_list():
    train_annotations = read_annotation(TRAIN_ANNOTATION_LIST_PATH)
    train_classes = set(train_annotations.values())
    eval_annotations = read_annotation(EVAL_ANNOTATION_LIST_PATH)
    eval_classes = set(eval_annotations.values())
    total_classes = train_classes.union(eval_classes)
    total_classes = sorted(list(total_classes))
    with open(MAPILLARY_CLASS_LIST_PATH, "w") as f:
        for cls in total_classes:
            f.write(f"{cls}\n")


@lru_cache
def mapillary_class_list():
    if not os.path.isfile(MAPILLARY_CLASS_LIST_PATH):
        write_mapillary_class_list()
    with open(MAPILLARY_CLASS_LIST_PATH) as f:
        class_list = [l.rstrip() for l in f.readlined()]
    return class_list


