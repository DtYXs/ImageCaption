import torch
import torchvision.transforms as transforms
# import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from pycocotools.coco import COCO
import random
import pickle

from dataset.Flickr import Flickr8k
from vocab import Vocabulary


def collate_fn_Flickr8k(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption).
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    images, captions_list = zip(*data)
    captions = [temp[random.randint(0, 4)] for temp in captions_list]  #从5个caption中随机选1个
    # Sort a data list by caption length (descending order).
    data = list(zip(images, captions))
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)
    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)  # dtyxs 第一维拼接成batch

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = torch.Tensor(cap[:end])
    return images, targets, lengths




def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption).
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions_list = zip(*data)
    captions = [caps[random.randint(0, len(caps)-1)] for caps in captions_list]  # 从5个caption中随机选1个

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)  # 第一维拼接成batch

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()  # 0为<PAD>
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return images, targets, lengths


def get_loader(dataset_name, images_root, json_file_path, vocab, transform, batch_size, shuffle, num_workers=1, is_train=True, is_val=False, is_test=False):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    assert dataset_name in ['Flickr8k', 'Flickr30k', 'COCO'], 'dataset不存在'

    if dataset_name == 'Flickr8k':
        dataset = Flickr8k(images_root=images_root,
                           json_file_path=json_file_path,
                           vocab=vocab,
                           is_train=is_train,
                           is_val=is_val,
                           is_test=is_test,
                           transform=transform
                           )

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn_Flickr8k)
    return data_loader
