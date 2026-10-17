"""
CUB-200-2011 metadata parsing and the PyTorch Dataset.

CUB ships its own flat metadata files (no taxonomy, just class ids):
  images.txt              image_id -> relative path
  image_class_labels.txt  image_id -> class_id (1-200)
  train_test_split.txt    image_id -> 1 (train) or 0 (test)
  bounding_boxes.txt      image_id -> x, y, width, height
  classes.txt             class_id -> class name (e.g. "001.Black_footed_Albatross")
"""

import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_cub_metadata(root: str) -> dict:
    """
    root is the extracted CUB_200_2011/ directory. Returns a dict with:
      images: image_id -> relative path (under images/)
      labels: image_id -> 0-indexed class id
      is_train: image_id -> bool
      bboxes: image_id -> (x, y, w, h) floats
      class_names: 0-indexed class id -> name string
    """
    images = {}
    with open(os.path.join(root, "images.txt")) as f:
        for line in f:
            img_id, path = line.strip().split()
            images[int(img_id)] = path

    labels = {}
    with open(os.path.join(root, "image_class_labels.txt")) as f:
        for line in f:
            img_id, cls_id = line.strip().split()
            labels[int(img_id)] = int(cls_id) - 1  # -> 0-indexed

    is_train = {}
    with open(os.path.join(root, "train_test_split.txt")) as f:
        for line in f:
            img_id, flag = line.strip().split()
            is_train[int(img_id)] = flag == "1"

    bboxes = {}
    with open(os.path.join(root, "bounding_boxes.txt")) as f:
        for line in f:
            img_id, x, y, w, h = line.strip().split()
            bboxes[int(img_id)] = (float(x), float(y), float(w), float(h))

    class_names = {}
    with open(os.path.join(root, "classes.txt")) as f:
        for line in f:
            cls_id, name = line.strip().split(None, 1)
            class_names[int(cls_id) - 1] = name  # -> 0-indexed

    return {
        "images": images,
        "labels": labels,
        "is_train": is_train,
        "bboxes": bboxes,
        "class_names": class_names,
    }


class CUBDataset(Dataset):
    """
    One sample = one bird, cropped to CUB's own bounding box (not the full
    photo -- the bounding box crop is what the paper's detector pipeline
    would see post-localisation, and it removes most background clutter,
    which matters for a fine-grained task).
    """

    def __init__(self, root: str, meta: dict, train: bool):
        self.root = root
        self.image_ids = [i for i, t in meta["is_train"].items() if t == train]
        self.images = meta["images"]
        self.labels = meta["labels"]
        self.bboxes = meta["bboxes"]

        if train:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.root, "images", self.images[img_id])
        image = Image.open(img_path).convert("RGB")

        x, y, w, h = self.bboxes[img_id]
        image = image.crop((x, y, x + w, y + h))

        image = self.transform(image)
        label = self.labels[img_id]
        return image, label
