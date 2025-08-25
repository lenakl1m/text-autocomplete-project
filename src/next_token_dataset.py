import torch
from torch.utils.data import Dataset
import pandas as pd
from collections import Counter


class NextTokenDataset(Dataset):
    # датасет для задачи предсказания следующего токена
    # ожидает csv с колонкой tokens, где каждая строка - список токенов

    def __init__(self, csv_path, 
                 max_vocab_size = 5000, 
                 min_freq = 2, 
                 vocab=None):
        # csv_path: путь к train.csv / val.csv / test.csv
        # max_vocab_size: максимальный размер словаря
        # min_freq: минимальная частота слова, чтобы попасть в словарь
        # vocab: словарь, если пустой — строится по данным

        df = pd.read_csv(csv_path)

        # преобразуем строку в список, если нужно
        if isinstance(df['tokens'].iloc[0], str):
            self.sentences = df['tokens'].apply(eval).tolist()
        else:
            self.sentences = df['tokens'].tolist()

        # строим или принимаем словарь
        if vocab is None:
            self.vocab = self._build_vocab(self.sentences, max_vocab_size, min_freq)
        else:
            self.vocab = vocab

        # кодируем все предложения
        self.data = [self._encode(sent) for sent in self.sentences]

    def _build_vocab(self, sentences, max_vocab_size, min_freq):
        # строит словарь по частоте слов
        counter = Counter()
        for sent in sentences:
            counter.update(sent)

        vocab_list = ['<pad>', '<unk>']
        vocab_list += [
            word for word, freq in counter.most_common()
            if freq >= min_freq and word not in vocab_list
        ]
        # ограничение по размеру
        vocab_list = vocab_list[:max_vocab_size]

        return {word: idx for idx, word in enumerate(vocab_list)}

    def _encode(self, tokens):
        # преобразует токены в индексы
        return [
            self.vocab.get(token, self.vocab['<unk>'])
            for token in tokens
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # аозвращает пару x y
        encoded = self.data[idx]
        if len(encoded) < 2:
            # пропускаем слишком короткие последовательности
            return self.__getitem__((idx + 1) % len(self.data))

        x = encoded[:-1]  # всё, кроме последнего
        y = encoded[1:]   # всё, кроме первого

        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

    def get_vocab_size(self):
        # возвращает размер словар
        return len(self.vocab)

    def get_vocab(self):
        # возвращает словарь
        return self.vocab

    def decode(self, indices):
        # декодирует индексы обратно в слова для интерпретации предсказаний
        inv_vocab = {idx: word for word, idx in self.vocab.items()}
        return [inv_vocab.get(idx, '<unk>') for idx in indices]