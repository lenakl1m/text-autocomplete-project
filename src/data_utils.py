import os
import re
import html
import random
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from src.token_dataset import AutoregressiveTokenDataset


def load_tweets_from_text(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        texts = [line.strip() for line in lines if line.strip()]
        print(f"загружено {len(texts)} строк из файла")
        return texts
    except FileNotFoundError:
        print(f"файл {file_path} не найден!")
        return []
    except Exception as e:
        print(f"ошибка при загрузке файла: {e}")
        return []


def clean_tweet(text):
    if not text or not isinstance(text, str):
        return ""
    
    text = html.unescape(text)
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+|\S+\.(com|ru|org|net|edu|gov|info)\S*', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^a-z0-9\s\']', ' ', text)
    
    replacements = {
        r"\bdon t\b": "don't", r"\bdidn t\b": "didn't", r"\bwon t\b": "won't",
        r"\bcant\b": "can't", r"\bwouldnt\b": "wouldn't", r"\bshouldnt\b": "shouldn't",
        r"\bwasnt\b": "wasn't", r"\bwere nt\b": "weren't", r"\bhas nt\b": "hasn't",
        r"\bhavent\b": "haven't", r"\bhad nt\b": "hadn't", r"\bdoes nt\b": "doesn't",
        r"\bcould nt\b": "couldn't", r"\bshould nt\b": "shouldn't", r"\bwill not\b": "won't",
        r"\bdo not\b": "don't", r"\bcan not\b": "can't", r"\bwould not\b": "wouldn't",
        r"\bshould not\b": "shouldn't", r"\bhave not\b": "haven't", r"\bhas not\b": "hasn't",
        r"\bhad not\b": "hadn't", r"\bdoes not\b": "doesn't", r"\bcould not\b": "couldn't",
        r"\bdint\b": "didn't", r"\bdidnt\b": "didn't", r"\bwont\b": "won't",
        r"\bdoesnt\b": "doesn't", r"\bshouldnt\b": "shouldn't", r"\bwasnt\b": "wasn't",
        r"\bwerent\b": "weren't", r"\bhavent\b": "haven't", r"\bhadnt\b": "hadn't",
        r"\bhasnt\b": "hasn't"
    }
    
    for pattern, repl in replacements.items():
        text = re.sub(pattern, repl, text)

    text = re.sub(r"(?<!n)'(?!t\b)", " ", text)
    text = re.sub(r"'{2,}", " ", text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def prepare_data(file_path, sample_ratio=0.2, min_len=4, max_len=16, seq_len=10, random_seed=42):
    print("загружаем и подготавливаем данные")
    print("-" * 40)
    random.seed(random_seed)
    np.random.seed(random_seed)

    texts = load_tweets_from_text(file_path)
    print(f'всего строк в файле: {len(texts)}')

    if not texts:
        print("данные не загружены, создаем демонстрационные данные")
        texts = [
            "I love to play football with my friends",
            "Today is a beautiful day for a walk",
            "I can't believe how fast time flies",
            "You should try this amazing recipe",
            "We are going to the beach tomorrow"
        ]

    random.shuffle(texts)
    sample_size = int(len(texts) * sample_ratio)
    texts = texts[:sample_size]
    print(f'оставлено после сэмплирования: {len(texts)} строк')

    cleaned_texts = [clean_tweet(t) for t in texts if t and t.strip()]
    print(f'очищено текстов: {len(cleaned_texts)}')
    print("-" * 40)

    print('\nпримеры очистки:')
    for i in range(min(5, len(texts))):
        print(f'до:  {texts[i]}')
        print(f'после: {cleaned_texts[i]}\n')
        print("-" * 40)

    filtered_texts = []
    for t in cleaned_texts:
        words = t.split()
        if min_len <= len(words) <= max_len:
            filtered_texts.append(t)
    print(f'после фильтрации по длине [{min_len}–{max_len}]: {len(filtered_texts)} текстов')

    lengths = [len(t.split()) for t in filtered_texts]
    if lengths:
        print(f'средняя длина: {np.mean(lengths):.1f} слов')
        print(f'мин: {min(lengths)}, макс: {max(lengths)}')

    train_texts, temp_texts = train_test_split(
        filtered_texts, test_size=0.2, random_state=random_seed, shuffle=True
    )
    val_texts, test_texts = train_test_split(
        temp_texts, test_size=0.5, random_state=random_seed, shuffle=True
    )

    print("-" * 40)
    print(f'train: {len(train_texts)}')
    print(f'val:   {len(val_texts)}')
    print(f'test:  {len(test_texts)}')
    print("-" * 40)

    all_tokens = [word for text in train_texts for word in text.split()]
    vocab = ['<PAD>', '<UNK>'] + list(set(all_tokens))
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    vocab_size = len(vocab)

    print(f'размер словаря: {vocab_size}')
    print(f"примеры: 'love' -> {word_to_idx.get('love', 'не найдено')}, 'the' -> {word_to_idx.get('the', 'не найдено')}")
    print("-" * 40)

    return train_texts, val_texts, test_texts, word_to_idx, idx_to_word, vocab_size, seq_len


def create_data_loaders(train_texts, val_texts, test_texts, word_to_idx, seq_len, batch_size=128):
    print("создаём датасеты и даталоадеры")
    
    train_dataset = AutoregressiveTokenDataset(train_texts, word_to_idx, seq_len)
    val_dataset = AutoregressiveTokenDataset(val_texts, word_to_idx, seq_len)
    test_dataset = AutoregressiveTokenDataset(test_texts, word_to_idx, seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    print("-" * 40)
    print(f'размер train_dataset: {len(train_dataset)} пар (контекст → следующий токен)')
    print("-" * 40)
    print(f'пример из train_dataset[0]: input={train_dataset[0][0]}, target={train_dataset[0][1]}')
    print("-" * 40)

    x_batch, y_batch = next(iter(train_loader))
    print(f'x_batch.shape: {x_batch.shape}  # [b, seq_len]')
    print(f'y_batch.shape: {y_batch.shape}  # [b]')
    print("-" * 40)

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset


def save_artifacts(word_to_idx, idx_to_word, vocab_size, seq_len, train_dataset, val_dataset, test_dataset):
    print("сохраняем артефакты")
    
    try:
        with open('data/vocab_final.pkl', 'wb') as f:
            pickle.dump({
                'word_to_idx': word_to_idx,
                'idx_to_word': idx_to_word,
                'vocab_size': vocab_size,
                'seq_len': seq_len
            }, f)
        torch.save(train_dataset, 'data/train_dataset_final.pt')
        torch.save(val_dataset, 'data/val_dataset_final.pt')
        torch.save(test_dataset, 'data/test_dataset_final.pt')
        print('датасет готов, словарь сохранён')
    except Exception as e:
        print(f"ошибка при сохранении артефактов: {e}")