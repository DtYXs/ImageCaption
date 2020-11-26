from collections import defaultdict
from PIL import Image

import os
import json
import torch


class Flickr8k(torch.utils.data.Dataset):
    def __init__(self, images_root, json_file_path, vocab, is_train=True, is_val=False, is_test=False, transform=None):
        self.images_root = os.path.expanduser(images_root)
        self.json_file_path = os.path.expanduser(json_file_path)
        self.vocab = vocab
        self.is_train, self.is_val, self.is_test = is_train, is_val, is_test
        if (self.is_train or self.is_val or self.is_test) is False:
            raise ValueError('Dataset must be train, val or test !')
        self.transform = transform

        # 读取json
        with open(self.json_file_path, 'r') as f:
            datas = json.load(f)['images']  # list 8000

        self.captions_raw = defaultdict(list)  # 存储raw caption
        self.captions_tokens = defaultdict(list)  # 存储tokens caption
        self.captions = defaultdict(list)  # 存储添加<START>，<END>的tokens
        self.train_ids, self.val_ids, self.test_ids = [], [], []
        for data in datas:
            img_id = data['filename']
            for s in data['sentences']:
                self.captions_raw[img_id].append(s['raw'])  # list
                self.captions_tokens[img_id].append(s['tokens'])  # list
                self.captions[img_id].append([])
                self.captions[img_id][-1].append(vocab('<START>'))
                self.captions[img_id][-1].extend([vocab(token) for token in self.captions_tokens[img_id][-1]])
                self.captions[img_id][-1].append(vocab('<END>'))

            if data['split'] == 'train':
                self.train_ids.append(img_id)
            elif data['split'] == 'val':
                self.val_ids.append(img_id)
            elif data['split'] == 'test':
                self.test_ids.append(img_id)
            else:
                raise ValueError('data[\'split\'] must be train, val or test !')

        self.train_ids.sort()
        self.val_ids.sort()
        self.test_ids.sort()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is a list of captions for the image.
        """
        if self.is_val:
            img_id = self.val_ids[index]
        elif self.is_test:
            img_id = self.test_ids[index]
        elif self.is_train:
            img_id = self.train_ids[index]
        else:
            raise ValueError

        # Image
        filename = os.path.join(self.images_root, img_id)
        img = Image.open(filename).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        # Captions
        # if self.target_transform is not None:  # 对raw caption操作
        #     target = self.captions_raw[img_id]
        #     target = self.target_transform(target)
        # else:
        #     target = (self.captions_tokens[img_id])  # tokens caption
        caption = self.captions[img_id]  # list
        '''
        [
         [1, 4, 477, 566, 6, 63, 16, 315, 31, 2],
         [1, 4, 74, 6, 63, 16, 31, 2],
         [1, 74, 15, 6, 63, 11, 315, 31, 2],
         [1, 318, 15, 6, 63, 16, 31, 2],
         [1, 283, 6, 70, 11, 315, 31, 2]
        ]
        '''
        return img, caption

    def __len__(self):
        if self.is_train:
            return len(self.train_ids)
        elif self.is_val:
            return len(self.val_ids)
        elif self.is_test:
            return len(self.test_ids)


