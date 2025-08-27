# import os
import pandas as pd
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from typing import List
from collections import Counter
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from src.next_token_dataset import NextTokenDataset

# качаем ресурсы для nltk
nltk.download('punkt_tab', quiet=True)

# очистка (оставляем только слова и пробелы)
def clean_text(text: str) -> str:
    # нижний регистр
    text = text.lower()
    
    # упоминания (@username)
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    
    # ссылки + почту
    text = re.sub(r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?', '', text)
    text = re.sub(r'www\.(?:[-\w.])+(?:/(?:[\w/_.])*)?(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?', '', text)
    text = re.sub(r'\b(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,}\b', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    
    # эмодзи и не ascii символы
    text = re.sub(r'[' + string.punctuation + ']', ' ', text)
    
    # пунктуация 
    text = re.sub(r'[' + string.punctuation + ']', ' ', text)
    
    # лишние пробелы
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def tokenize_text(text: str) -> List[str]:
    # токенизируем текст с помощью nlkt
    return word_tokenize(text)

def load_and_clean_dataset(raw_path: str, processed_path: str):
#     # проверяем, существует ли уже обработанный файл
#     if os.path.exists(processed_path):
#         print(f"{processed_path} уже существует, загружаю...")
#         return pd.read_csv(processed_path)

    # загружаем сырой датасет, очищаем и сохраняем, построчно
    with open(raw_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    cleaned_texts = []
    for line in lines:
        cleaned = clean_text(line)
        if cleaned:  # после очистки текст не пустой
            cleaned_texts.append(cleaned)

    # датафрейм
    df = pd.DataFrame({'text': cleaned_texts})

    # токенезируем
    df['tokens'] = df['text'].apply(tokenize_text)

   # cохраняем
    df.to_csv(processed_path, index=False)
    print(f"Обработанный датасет сохранён в {processed_path}")
    print(f"\nКоличество текстов: {len(df)}")
    return df

# анализ уже сделанного словаря, уникальные слова + покрытие текста
def analyze_vocab(df):
    counter = Counter()
    for tokens in df['tokens']:
        counter.update(tokens)
    
    print(f"Всего вхождений: {sum(counter.values())}")
    print(f"Уникальных токенов: {len(counter)}")
    
    singletons = sum(1 for freq in counter.values() if freq == 1)
    print(f"Слов с частотой 1: {singletons}")

    cumsum = 0
    total = sum(counter.values())
    for i, (word, freq) in enumerate(counter.most_common()):
        cumsum += freq
        if cumsum / total > 0.90 and i < 1000:
            print(f"Топ-{i+1} слов покрывают 90%")
        if cumsum / total > 0.95 and i < 1000:
            print(f"Топ-{i+1} слов покрывают 95%")
        if cumsum / total > 0.99:
            print(f"Топ-{i+1} слов покрывают 99%")
            break

# разбивка на train (80%), val (10%), test (10%)

def split_dataset(df, train_path, val_path, test_path):
    train, temp = train_test_split(df, test_size=0.2, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)
    
    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)
    test.to_csv(test_path, index=False)
    
    print(f"Разделение:")
    print(f"\tТрейн: {len(train)}")
    print(f"\tВалидация: {len(val)}")
    print(f"\tТест: {len(test)}")

    return train, val, test

def collate_fn(batch):
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    # паддинг до максимальной длины в батче
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=-100)
    
    return inputs_padded, targets_padded

def get_data_loaders(train_path, val_path, test_path, batch_size, max_vocab_size, min_freq):

    # датасеты 
    # nренировочный строит словарь
    train_dataset = NextTokenDataset(
        csv_path=train_path,
        max_vocab_size=max_vocab_size,
        min_freq=min_freq
    )

    vocab = train_dataset.get_vocab()

    # валидационный и тестовый используют готовый словарь
    val_dataset = NextTokenDataset(
        csv_path=val_path,
        vocab=vocab
    )
    
    test_dataset = NextTokenDataset(
        csv_path=test_path,
        vocab=vocab
    )

    # даталоадеры
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader, train_dataset.get_vocab()