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
from vocab.Vocabulary import Vocabulary


def collate_fn_Flickr8k(data):
    images, captions_list = zip(*data)
    captions = [temp[random.randint(0, 4)] for temp in captions_list]
    data = list(zip(images, captions))
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)
    images = torch.stack(images, 0)

    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = torch.Tensor(cap[:end])
    return images, targets, lengths


def collate_fn(data):
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions_list = zip(*data)
    captions = [caps[random.randint(0, len(caps) - 1)] for caps in captions_list]
    images = torch.stack(images, 0)

    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return images, targets, lengths


def get_loader(dataset_name, images_root, json_file_path, vocab, transform, batch_size, shuffle, num_workers=1,
               is_train=True, is_val=False, is_test=False):
    assert dataset_name in ['Flickr8k', 'Flickr30k', 'COCO'], 'dataset does not exist.'

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


if __name__ == '__main__':
    args = {}
    args['dataset'] = 'Flickr8k'
    args['vocab_word2idx_path'] = '../vocab/save/' + args['dataset'] + '/vocab/thre5_word2idx.pkl'
    args['vocab_idx2word_path'] = '../vocab/save/' + args['dataset'] + '/vocab/thre5_idx2word.pkl'
    args['vocab_idx_path'] = '../vocab/save/' + args['dataset'] + '/vocab/thre5_idx.pkl'
    args['crop_size'] = 224
    args['batch_size'] = 128
    args['embed_size'] = 256
    args['hidden_size'] = 512
    args['epoch'] = 10
    args['save_step'] = 1
    args['model_save_root'] = '../save/'
    vocab = Vocabulary()
    with open(args['vocab_word2idx_path'], 'rb') as f:
        vocab.word2idx = pickle.load(f)
    with open(args['vocab_idx2word_path'], 'rb') as f:
        vocab.idx2word = pickle.load(f)
    with open(args['vocab_idx_path'], 'rb') as f:
        vocab.idx = pickle.load(f)
    transform = transforms.Compose([
        transforms.RandomCrop(args['crop_size']),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4580, 0.4464, 0.4032),
                             (0.2318, 0.2229, 0.2269))
    ])
    loader_train = get_loader(dataset_name='Flickr8k',
                              images_root='./data/Flickr/8k/Flicker8k_Dataset_resize256x256',
                              json_file_path='./data/Flickr/8k/dataset_flickr8k.json',
                              vocab=vocab,
                              transform=transform,
                              batch_size=args['batch_size'],
                              shuffle=True,
                              is_train=True
                              )

    print(1)