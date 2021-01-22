import os
import json
import pickle
import argparse
from collections import Counter

from vocab.Vocabulary import Vocabulary


def build_vocab(dataset_name, json_file_path, threshold=0):
    """
    生成词汇表vocab.pkl文件，生成数据集的word2idx.txt和word_frequency.txt文件

    Args:
        dataset_name(str): Flickr8k, Flickr30k or COCO
        threshold(int): 过滤频率小于threshold的单词
    """
    assert dataset_name in ['Flickr8k', 'Flickr30k', 'COCO'], 'dataset不存在'

    if dataset_name == 'Flickr8k':
        with open(json_file_path, 'r') as f:
            datas = json.load(f)['images']  # list 8000

        i = 0
        counter = Counter()
        for data in datas:
            for s in data['sentences']:
                i += 1
                counter.update(s['tokens'])

                if i % 10000 == 0:
                    print("[{}/40000] Tokenized the captions.".format(i))

    vocab = Vocabulary()
    vocab.add_word('<PAD>')
    vocab.add_word('<START>')
    vocab.add_word('<END>')
    vocab.add_word('<UNK>')
    words = [word for word, cnt in counter.items() if cnt >= threshold]
    for word in words:
        vocab.add_word(word)
    words_frequency_order = sorted(counter.items(), key=lambda x: x[1], reverse=True)

    # 写入txt文件
    if not os.path.exists('./save/' + dataset_name + '/' + dataset_name + '_thre' + str(threshold) + '_word2idx.txt'):
        with open('./save/' + dataset_name + '/' + dataset_name + '_thre' + str(threshold) + '_word2idx.txt', 'w', encoding='utf-8') as f:
            for i in range(vocab.idx):
                f.write(vocab.idx2word[i] + ' ' + str(i))
                f.write('\n')
    if not os.path.exists('./save/' + dataset_name + '/' + dataset_name + '_words_frequency.txt'):
        with open('./save/' + dataset_name + '/' + dataset_name + '_words_frequency.txt', 'w', encoding='utf-8') as f:
            for (word, idx) in words_frequency_order:
                f.write(word + ' ' + str(idx))
                f.write('\n')

    # 写入vocab.pkl文件\
    if not os.path.exists('./save/' + dataset_name + '/vocab/' + 'thre' + str(threshold) + '_word2idx.pkl'):
        with open('./save/' + dataset_name + '/vocab/' + 'thre' + str(threshold) + '_word2idx.pkl', 'wb') as f:
            pickle.dump(vocab.word2idx, f)
        with open('./save/' + dataset_name + '/vocab/' + 'thre' + str(threshold) + '_idx2word.pkl', 'wb') as f:
            pickle.dump(vocab.idx2word, f)
        with open('./save/' + dataset_name + '/vocab/' + 'thre' + str(threshold) + '_idx.pkl', 'wb') as f:
            pickle.dump(vocab.idx, f)

    return counter


def main(args):
    counter = build_vocab(dataset_name=args.dataset_name,
                        # images_root=args.images_root,
                        json_file_path=args.json_file_path,
                        threshold=args.threshold
                        )

    print("Total vocabulary size: {}".format(len(counter)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='Flickr8k',
                        help='数据集名称')
    # parser.add_argument('--images_root', type=str, default='E:/DL/Flickr/8k/Flicker8k_Dataset/',
    #                     help='数据集图像根目录')
    parser.add_argument('--json_file_path', type=str, default='./data/Flickr/8k/dataset_flickr8k.json',
                        help='数据集json文件')
    parser.add_argument('--threshold', type=int, default=5,
                        help='过滤单词频率阈值')
    args = parser.parse_args()
    main(args)
