"""
*********************************************
* @Project ：NIC 
* @File    ：utls.py
* @Author  ：DtYXs
* @Date    ：2021/1/17 20:14 
*********************************************
"""
import json
import copy
import os
import pickle
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm

from model import *
from dataset.data_loader import get_loader
from vocab.Vocabulary import Vocabulary


def cut_dataset():
    """
    分割数据集
    """
    with open('./data/COCO/annotations/captions_train2014.json', "r", encoding='utf-8') as f:
        datas = json.loads(f.read())
        out_datas = copy.deepcopy(datas)
        out_datas['images'] = []
        out_datas['annotations'] = []

        for i in tqdm(range(100)):
            id = datas['images'][i]['id']
            out_datas['images'].append(datas['images'][i])
            for an in datas['annotations']:
                if an['image_id'] == id:
                    out_datas['annotations'].append(an)

        with open('data.json', 'w') as f:
            json.dump(out_datas, f)


def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False