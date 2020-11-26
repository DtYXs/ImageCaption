class Vocabulary(object):
    """
    单词包装器
    """
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not (word in self.word2idx):
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word_or_idx):
        if isinstance(word_or_idx, str):
            if not (word_or_idx) in self.word2idx:
                return self.word2idx['<UNK>']
            return self.word2idx[word_or_idx]
        if isinstance(word_or_idx, int):
            assert 0 <= word_or_idx <= self.idx
            return self.idx2word[word_or_idx]
        raise ValueError('输入必须是int或str')

    def __len__(self):
        return len(self.word2idx)
