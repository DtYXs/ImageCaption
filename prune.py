"""
*********************************************
* @Project ：NIC 
* @File    ：prune.py
* @Author  ：DtYXs
* @Date    ：2021/1/17 20:08 
*********************************************
"""
import json
import copy
from tqdm import tqdm
import os
import pickle
import time
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from pycocoevalcap.bleu.bleu import Bleu
from tqdm import tqdm

from model import *
from dataset.data_loader import get_loader
from vocab.Vocabulary import Vocabulary
from utls import set_random_seed


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = {}
    config['cut'] = True
    config['dataset'] = 'COCO'
    config['vocab_word2idx_path'] = './vocab/save/' + 'COCO' + '/vocab/' + 'thre5_word2idx.pkl'
    config['vocab_idx2word_path'] = './vocab/save/' + 'COCO' + '/vocab/' + 'thre5_idx2word.pkl'
    config['vocab_idx_path'] = './vocab/save/' + 'COCO' + '/vocab/' + 'thre5_idx.pkl'
    config['crop_size'] = 224
    config['images_root_train'] = './data/COCO/train2014_resized'
    config['images_root_val'] = './data/COCO/val2014_resized'
    config['json_file_path_train'] = './data/COCO/annotations/captions_train2014.json'
    config['json_file_path_val'] = './data/COCO/annotations/captions_val2014.json'
    config['batch_size'] = 56
    config['embed_size'] = 256
    config['hidden_size'] = 512
    config['learning_rate'] = 5e-5
    config['epoch_num'] = 0
    config['save_step'] = 1
    config['model_save_root'] = './save/' if (config['cut'] is True) else './save/cut/'

    config['prune_percent'] = 0.5
    config['encoder_path'] = './save/'
    config['decoder_path'] = './save/'
    config['encoder_pruned_path'] = './save/'
    config['decoder_pruned_path'] = './save/'

    set_random_seed(109)


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
    # 获得要剪枝层的序号
    ####################################################
    encoder_ori = EncoderCNN(config['embed_size'])
    encoder_ori.load_state_dict(torch.load(config['encoder_path']))
    bn_nums = []
    for i, m in enumerate(encoder_ori.vgg.modules()):
        if isinstance(m, nn.BatchNorm2d):
            bn_nums.append(i)
            print(str(i) + ' ' + str(m))
    for i, m in enumerate(encoder_ori.vgg.modules()):
        if isinstance(m, nn.Conv2d):
            print(str(i) + ' ' + str(m))

    bn_nums = [3, 6, 10, 13, 17, 20, 23, 27, 30, 33, 37, 40, 43]
    bn_prune_nums = bn_nums

    total_gamma_num = 0
    for m in encoder_ori.vgg.modules():
        if isinstance(m, nn.BatchNorm2d):
            total_gamma_num += m.weight.data.shape[0]

    bn_gammas = torch.zeros(total_gamma_num)
    i = 0
    for m in encoder_ori.vgg.modules():
        if isinstance(m, nn.BatchNorm2d):
            size = m.weight.data.shape[0]
            bn_gammas[i: (i + size)] = m.weight.data.abs().clone()
            i += size

    bn_gammas_sorted, _ = torch.sort(bn_gammas)
    gamma_prune_percent = config['prune_percent']
    gamma_thre = bn_gammas_sorted[int((total_gamma_num - 1) * gamma_prune_percent)]
    print(gamma_thre)

    masks = []
    layer_cfg = []
    prune_channel_nums = []
    for i, m in enumerate(encoder_ori.vgg.modules()):
        if isinstance(m, nn.BatchNorm2d) and (i in bn_prune_nums):
            weight_copy = m.weight.data.abs().clone()
            mask = weight_copy.gt(gamma_thre).float()
            remaining_channel_num = int(torch.sum(mask).item())
            layer_cfg.append(remaining_channel_num)
            prune_channel_num = int(mask.shape[0]) - remaining_channel_num
            prune_channel_nums.append(prune_channel_num)
            print('BN:%d 低于阈值gamma比例:%.4f%% 剪掉通道数:%d 剩余通道数:%d' %
                  (i, (1.0-torch.sum(mask).item()/mask.shape[0])*100, prune_channel_num, remaining_channel_num))
            mask = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())), axis=1)
            if mask.size == 1:
                mask = np.resize(mask, (1, ))
            masks.append(mask)
        elif isinstance(m, nn.MaxPool2d):
            layer_cfg.append('M')
    masks = [mask.tolist() for mask in masks]


    ####################################################
    # 生成剪枝后模型
    ####################################################
    layer_cfg[-2] = 512
    encoder_pruned = EncoderCNN_prune(config['embed_size'], layer_cfg=layer_cfg)
    i = 0
    start_mask = [0, 1, 2]
    end_mask = masks[i]
    for (m0, m1) in zip(encoder_ori.modules(), encoder_pruned.modules()):
        if isinstance(m0, nn.Conv2d):
            # print('do conv   %d' % i)
            if i < len(masks) - 1:
                w1 = m0.weight.data[:, start_mask, :, :].clone()
                w1 = w1[end_mask, :, :, :].clone()
                m1.weight.data = w1.clone()
                m1.bias.data = m0.bias.data[end_mask].clone()
            elif i == len(masks) - 1:
                m1.weight.data = m0.weight.data[:, start_mask, :, :].clone()
                m1.bias.data = m0.bias.data.clone()
        elif isinstance(m0, nn.BatchNorm2d):
            # print('do bn   %d' % i)
            if i < len(masks) - 1:
                m1.weight.data = m0.weight.data[end_mask].clone()
                m1.bias.data = m0.bias.data[end_mask].clone()
                m1.running_mean = m0.running_mean[end_mask].clone()
                m1.running_var = m0.running_var[end_mask].clone()
                i += 1
                start_mask = copy.deepcopy(end_mask)
                end_mask = masks[i]
            elif i == len(masks) - 1:
                m1.weight.data = m0.weight.data.clone()
                m1.bias.data = m0.bias.data.clone()
                m1.running_mean = m0.running_mean.clone()
                m1.running_var = m0.running_var.clone()
        elif isinstance(m0, nn.Linear):
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()


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
                              images_root=config['images_root_train'],
                              json_file_path=config['json_file_path_train'],
                              vocab=vocab,
                              transform=transform,
                              batch_size=config['batch_size'],
                              shuffle=True,
                              is_train=True
                              )
    loader_val = get_loader(dataset_name=config['dataset'],
                            images_root=config['images_root_val'],
                            json_file_path=config['json_file_path_val'],
                            vocab=vocab,
                            transform=transform,
                            batch_size=1,
                            shuffle=False,
                            is_val=True
                            )


    ####################################################
    # create pruned model
    ####################################################
    encoder_ori = encoder_ori.to(device)
    decoder_ori = DecoderRNN(config['embed_size'], config['hidden_size'], len(vocab), 1).to(device)
    decoder_ori.load_state_dict(torch.load(config['decoder_path']))
    encoder_pruned = encoder_pruned.to(device)
    encoder_pruned.load_state_dict(torch.load(config['encoder_pruned_path']))
    decoder_pruned = DecoderRNN(config['embed_size'], config['hidden_size'], len(vocab), 1).to(device)
    decoder_pruned.load_state_dict(torch.load(config['decoder_pruned_path']))


    ####################################################
    # test pruned model
    ####################################################
    def test(encoder, decoder, loader_val):
        encoder.eval()
        decoder.eval()
        raw_captions = []
        sampled_captions = []
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
            if sampled_caption[-1] != '.':
                sampled_caption.append('.')
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

    test(encoder_ori, decoder_ori, loader_val)
    test(encoder_pruned, decoder_pruned, loader_val)

    ####################################################
    # retrain pruned model
    ####################################################
    # criterion = nn.CrossEntropyLoss()
    # params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    # # params = list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    # optimizer = torch.optim.Adam(params, lr=config['learning_rate'])
    # if config['cut'] is True:
    #     params_vgg = list(encoder.vgg.parameters())
    #     params = [{'params': params},
    #               {'params': params_vgg, 'lr': config['learning_rate'] * 0.1}]
    #     optimizer = torch.optim.Adam(params, lr=config['learning_rate'])
    #
    # total_step = len(loader_train)
    # # best_BLEU4_score = -1
    # for epoch in range(config['epoch_num']):
    #     encoder.train()
    #     decoder.train()
    #     for i, (images, captions, lengths) in enumerate(tqdm(loader_train)):
    #
    #         # Set mini-batch dataset
    #         images = images.to(device)
    #         captions = captions.to(device)
    #         targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
    #
    #         # Forward, backward and optimize
    #         features = encoder(images)
    #         outputs = decoder(features, captions, lengths)
    #         loss = criterion(outputs, targets)
    #         decoder.zero_grad()
    #         encoder.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #
    #         # Print log info
    #         if i % 10 == 0:
    #             print('Epoch [{}/{}], Iteration [{}/{}], Loss: {:.5f}, Perplexity: {:5.5f}'
    #                   .format(epoch, config['epoch_num'], i, total_step, loss.item(), np.exp(loss.item())))
    #
    #     # Save the model checkpoints
    #     if (epoch + 1) % config['save_step'] == 0:
    #         # Check BLEU score
    #
    #         # encoder_temp = EncoderCNN(config['embed_size']).to(device).eval()
    #         # decoder_temp = DecoderRNN(config['embed_size'], config['hidden_size'], len(vocab), 1).to(device)
    #         # encoder_temp.load_state_dict(torch.load('./save/' + config['dataset'] + 'encoder-{}.pth'.format(epoch+1)))
    #         # decoder_temp.load_state_dict(torch.load('./save/' + config['dataset'] + 'decoder-{}.pth'.format(epoch+1)))
    #         encoder.eval()
    #         decoder.eval()
    #         BLEU4_score = 0.
    #         for i, (image, caption, length) in tqdm(enumerate(loader_val)):
    #             image = image.to(device)
    #             feature = encoder(image)
    #             sampled_ids = decoder.sample(feature)
    #             sampled_ids = sampled_ids[0].cpu().numpy()
    #             sampled_caption = []
    #             for word_id in sampled_ids:
    #                 word = vocab.idx2word[word_id]
    #                 sampled_caption.append(word)
    #                 if word == '<END>':
    #                     break
    #             raw_caption = [[vocab(int(token)) for token in list(caption[0])]]
    #             sampled_caption = sampled_caption[1:-1]  # delete <START> and <END>
    #             raw_caption[0] = raw_caption[0][1:-1]  # delete <START> and <END>
    #             BLEU4_score += sentence_bleu(raw_caption, sampled_caption, weights=(0.25, 0.25, 0.25, 0.25),
    #                                          smoothing_function=SmoothingFunction().method1)
    #         BLEU4_score /= (i + 1)
    #         torch.save(encoder.state_dict(), os.path.join(
    #             config['model_save_root'], config['dataset'], 'encoder-{}-{:.5f}.pth'.format(epoch + 1, BLEU4_score)))
    #         torch.save(decoder.state_dict(), os.path.join(
    #             config['model_save_root'], config['dataset'], 'decoder-{}-{:.5f}.pth'.format(epoch + 1, BLEU4_score)))
    #         print(BLEU4_score)
    #     test(encoder, decoder, loader_val)
    #     print(1)



    from torchstat import stat
    encoder_cpu = copy.deepcopy(encoder_ori).to(device='cpu')
    encoder_pruned_cpu = copy.deepcopy(encoder_pruned).to(device='cpu')
    stat(encoder_cpu, (3, 224, 224))
    stat(encoder_pruned_cpu, (3, 224, 224))




if __name__ == '__main__':
    main()
    print('over')
