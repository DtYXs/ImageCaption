"""
*********************************************
* @Project ：NIC
* @File    ：test.py
* @Author  ：DtYXs
* @Date    ：2021/1/8 21:41
*********************************************
"""
import os
import pickle
import time
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
from pycocoevalcap.bleu.bleu import Bleu
from tqdm import tqdm

from model import EncoderCNN, DecoderRNN
from dataset.data_loader import get_loader
from vocab.Vocabulary import Vocabulary


def main():
    ####################################################
    # config
    ####################################################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = {}
    config['dataset'] = 'COCO'
    config['vocab_word2idx_path'] = './vocab/save/' + 'COCO' + '/vocab/' + 'thre5_word2idx.pkl'
    config['vocab_idx2word_path'] = './vocab/save/' + 'COCO' + '/vocab/' + 'thre5_idx2word.pkl'
    config['vocab_idx_path'] = './vocab/save/' + 'COCO' + '/vocab/' + 'thre5_idx.pkl'
    config['crop_size'] = 224
    config['images_root'] = './data/COCO/train2014_resized'
    config['json_file_path_train'] = './data/COCO/annotations/captions_mini100.json'
    config['json_file_path_val'] = './data/COCO/annotations/captions_val2014.json'
    config['batch_size'] = 128
    config['embed_size'] = 256
    config['hidden_size'] = 512
    config['learning_rate'] = 1e-4
    config['epoch_num'] = 20
    config['save_step'] = 10
    config['model_save_root'] = './save/'

    config['encoder_path'] = './save/'
    config['decoder_path'] = './save/'


    ####################################################
    # load vocabulary
    ####################################################
    vocab = Vocabulary()
    with open(config['vocab_word2idx_path'], 'rb') as f:
        vocab.word2idx = pickle.load(f)
    with open(config['vocab_idx2word_path'], 'rb') as f:
        vocab.idx2word = pickle.load(f)
    with open(config['vocab_idx_path'], 'rb') as f:
        vocab.idx = pickle.load(f)


    ####################################################
    # create data_loader
    ####################################################
    normalize = {
        'Flickr8k': [(0.4580, 0.4464, 0.4032),
                     (0.2318, 0.2229, 0.2269)],
        'Flickr30k': None,
        'COCO': [(0.485, 0.456, 0.406),
                 (0.229, 0.224, 0.225)]}

    transform = transforms.Compose([
        transforms.RandomCrop(config['crop_size']),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(normalize[config['dataset']][0],
                             normalize[config['dataset']][1])
    ])

    loader_train = get_loader(dataset_name=config['dataset'],
                              images_root=config['images_root'],
                              json_file_path=config['json_file_path_train'],
                              vocab=vocab,
                              transform=transform,
                              batch_size=config['batch_size'],
                              shuffle=True,
                              is_train=True
                              )
    loader_val = get_loader(dataset_name=config['dataset'],
                            images_root=config['images_root'],
                            json_file_path=config['json_file_path_val'],
                            vocab=vocab,
                            transform=transform,
                            batch_size=1,
                            shuffle=False,
                            is_val=True
                            )


    ####################################################
    # create model
    ####################################################
    encoder = EncoderCNN(config['embed_size'])
    decoder = DecoderRNN(config['embed_size'], config['hidden_size'], len(vocab), 1)
    encoder.load_state_dict(torch.load(config['encoder_path']))
    decoder.load_state_dict(torch.load(config['decoder_path']))
    encoder.to(device)
    decoder.to(device)

    ####################################################
    # create trainer
    ####################################################
    raw_captions = []
    sampled_captions = []

    encoder.eval()
    decoder.eval()
    for i, (image, caption, length) in enumerate(tqdm(loader_val)):
        image = image.to(device)
        feature = encoder(image)
        sampled_ids = decoder.sample(feature)
        sampled_ids = sampled_ids[0].cpu().numpy()
        sampled_caption = []
        for word_id in sampled_ids:
            word = vocab.idx2word[word_id]
            sampled_caption.append(word)
            if word == '<END>':
                break
        raw_caption = [[vocab(int(token)) for token in list(caption[0])]]
        sampled_caption = sampled_caption[1:-1]  # delete <START> and <END>
        # if sampled_caption[-1] != '.':
        #     sampled_caption.append('.')
        raw_caption[0] = raw_caption[0][1:-1]  # delete <START> and <END>
        raw_captions.append(raw_caption)
        sampled_captions.append(sampled_caption)

    hypo = {}
    for i, caption in enumerate(sampled_captions):
        hypo[i] = [' '.join(caption)]
    ref = {}
    for i, caption in enumerate(raw_captions):
        ref[i] = [' '.join(caption[0])]

    final_scores = Bleu().compute_score(ref, hypo)
    print(final_scores[0])

if __name__ == '__main__':
    main()
    print(1)
