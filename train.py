import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms

from model import EncoderCNN, DecoderRNN
from dataset.data_loader import get_loader
from vocab.Vocabulary import Vocabulary


def main():
    ####################################################
    # 参数配置
    ####################################################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = {}

    args['dataset'] = 'Flickr8k'
    args['vocab_word2idx_path'] = './vocab/save/' + args['dataset'] + '/vocab/thre5_word2idx.pkl'
    args['vocab_idx2word_path'] = './vocab/save/' + args['dataset'] + '/vocab/thre5_idx2word.pkl'
    args['vocab_idx_path'] = './vocab/save/' + args['dataset'] + '/vocab/thre5_idx.pkl'
    args['crop_size'] = 224
    args['batch_size'] = 128
    args['embed_size'] = 256
    args['hidden_size'] = 512
    args['epoch'] = 10
    args['save_step'] = 1
    args['model_save_root'] = './save/'


    ####################################################
    # 构建词表
    ####################################################
    vocab = Vocabulary()
    with open(args['vocab_word2idx_path'], 'rb') as f:
        vocab.word2idx = pickle.load(f)
    with open(args['vocab_idx2word_path'], 'rb') as f:
        vocab.idx2word = pickle.load(f)
    with open(args['vocab_idx_path'], 'rb') as f:
        vocab.idx = pickle.load(f)


    ####################################################
    # 创建data_loader
    ####################################################
    transform = transforms.Compose([
        transforms.RandomCrop(args['crop_size']),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4580, 0.4464, 0.4032),
                             (0.2318, 0.2229, 0.2269))
    ])

    loader_train = get_loader(dataset_name='Flickr8k',
                              images_root='E:/DL/Flickr/8k/Flicker8k_Dataset_resize256x256',
                              json_file_path='E:/DL/Flickr/8k/dataset_flickr8k.json',
                              vocab=vocab,
                              transform=transform,
                              batch_size=args['batch_size'],
                              shuffle=True,
                              is_train=True
                              )


    encoder = EncoderCNN(args['embed_size']).to(device)
    decoder = DecoderRNN(args['embed_size'], args['hidden_size'], len(vocab), 1).to(device)


    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3)

    total_step = len(loader_train)
    for epoch in range(args['epoch']):
        for i, (images, captions, lengths) in enumerate(loader_train):

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
            optimizer.step()

            # Print log info
            if i % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, args['epoch'], i, total_step, loss.item(), np.exp(loss.item())))

        # Save the model checkpoints
        if epoch % args['save_step'] == 0:
            torch.save(decoder.state_dict(), os.path.join(
                args['model_save_root'], 'decoder-{}-{}.ckpt'.format(epoch+1, i+1)))
            torch.save(encoder.state_dict(), os.path.join(
                args['model_save_root'], 'encoder-{}-{}.ckpt'.format(epoch+1, i+1)))



if __name__ == '__main__':
    main()











