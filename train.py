import os
import pickle
import time
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


def updateBN(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(1e-4 * torch.sign(m.weight.data))  # L1


def main():
    ####################################################
    # config
    ####################################################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = {}
    config['cut'] = True
    config['dataset'] = 'COCO'
    config['vocab_word2idx_path'] = './vocab/save/' + 'COCO' + '/vocab/' + 'thre5_word2idx.pkl'
    config['vocab_idx2word_path'] = './vocab/save/' + 'COCO' + '/vocab/' + 'thre5_idx2word.pkl'
    config['vocab_idx_path'] = './vocab/save/' + 'COCO' + '/vocab/' + 'thre5_idx.pkl'
    config['crop_size'] = 224
    config['images_root'] = 'E:/DL/COCO/train2014_resized'
    # 'E:/DL/Flickr/8k/Flicker8k_Dataset_resize256x256'
    #
    # 'E:/DL/COCO/val2014_resized'
    # 'E:/DL/COCO/train2014_resized'
    config['json_file_path_train'] = 'E:/DL/COCO/annotations/captions_train2014.json'
    # 'E:/DL/Flickr/8k/dataset_flickr8k.json'
    # 'E:/DL/COCO/annotations/captions_mini100.json'
    # 'E:/DL/COCO/annotations/captions_train2014.json'
    config['json_file_path_val'] = 'E:/DL/COCO/annotations/captions_val2014.json'
    # 'E:/DL/Flickr/8k/dataset_flickr8k.json'
    #
    # 'E:/DL/COCO/annotations/captions_val2014.json'
    config['batch_size'] = 18
    config['embed_size'] = 256
    config['hidden_size'] = 512
    config['learning_rate'] = 1e-4
    config['epoch_num'] = 20
    config['save_step'] = 1
    config['model_save_root'] = './save/' if (config['cut'] is True) else './save/cut/'


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
    encoder = EncoderCNN(config['embed_size']).to(device)
    if config['cut'] is True:
        encoder = EncoderCNN_cut(config['embed_size']).to(device)
    decoder = DecoderRNN(config['embed_size'], config['hidden_size'], len(vocab), 1).to(device)

    ####################################################
    # create trainer
    ####################################################
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=config['learning_rate'])
    if config['cut'] is True:
        params_vgg = list(encoder.vgg.parameters())
        params = [{'params': params},
                  {'params': params_vgg, 'lr': config['learning_rate'] * 0.1}]
        optimizer = torch.optim.Adam(params, lr=config['learning_rate'])

    total_step = len(loader_train)
    # best_BLEU4_score = -1
    for epoch in range(config['epoch_num']):
        encoder.train()
        decoder.train()
        for i, (images, captions, lengths) in enumerate(tqdm(loader_train)):

            # Set mini-batch dataset
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

            # Forward, backward and optimize
            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            loss = criterion(outputs, targets)
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            if config['cut'] is True:
                updateBN(encoder)
            optimizer.step()

            # Print log info
            if i % 10 == 0:
                print('Epoch [{}/{}], Iteration [{}/{}], Loss: {:.5f}, Perplexity: {:5.5f}'
                      .format(epoch, config['epoch_num'], i, total_step, loss.item(), np.exp(loss.item())))

        # Save the model checkpoints
        if (epoch + 1) % config['save_step'] == 0:
            # Check BLEU score

            # encoder_temp = EncoderCNN(config['embed_size']).to(device).eval()
            # decoder_temp = DecoderRNN(config['embed_size'], config['hidden_size'], len(vocab), 1).to(device)
            # encoder_temp.load_state_dict(torch.load('./save/' + config['dataset'] + 'encoder-{}.pth'.format(epoch+1)))
            # decoder_temp.load_state_dict(torch.load('./save/' + config['dataset'] + 'decoder-{}.pth'.format(epoch+1)))
            encoder.eval()
            decoder.eval()
            BLEU4_score = 0.
            for i, (image, caption, length) in tqdm(enumerate(loader_val)):
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
                raw_caption[0] = raw_caption[0][1:-1]  # delete <START> and <END>
                BLEU4_score += sentence_bleu(raw_caption, sampled_caption, weights=(0.25, 0.25, 0.25, 0.25),
                                             smoothing_function=SmoothingFunction().method1)
            BLEU4_score /= (i + 1)
            torch.save(encoder.state_dict(), os.path.join(
                config['model_save_root'], config['dataset'], 'encoder-{}-{:.5f}.pth'.format(epoch + 1, BLEU4_score)))
            torch.save(decoder.state_dict(), os.path.join(
                config['model_save_root'], config['dataset'], 'decoder-{}-{:.5f}.pth'.format(epoch + 1, BLEU4_score)))
            print(BLEU4_score)


            # if BLEU4_score > best_BLEU4_score:
            #     best_BLEU4_score = BLEU4_score
            #     torch.save(encoder.state_dict(),
            #                './save/' + config['dataset'] + '/best/' + 'encoder-{}.pth'.format(epoch + 1))
            #     torch.save(decoder.state_dict(),
            #                './save/' + config['dataset'] + '/best/' + 'decoder-{}.pth'.format(epoch + 1))


if __name__ == '__main__':
    main()
    print(1)

