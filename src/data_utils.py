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
    # загружаем строки из файла
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    texts = [line.strip() for line in lines if line.strip()]
    return texts


def clean_tweet(text):
    # html-сущности
    text = html.unescape(text)
    # нижний регистр
    text = text.lower()
    # ссылки
    text = re.sub(r'https?://\S+|www\.\S+|\S+\.(com|ru|org|net|edu|gov|info)\S*', '', text)
    # упоминания
    text = re.sub(r'@\w+', '', text)
    # хештеги
    text = re.sub(r'#\w+', '', text)
    # только буквы, цифры, пробел и апостроф
    text = re.sub(r'[^a-z0-9\s\']', ' ', text)
    # заменяем распространённые формы сокращений
    text = re.sub(r"\bdon t\b", "don't", text)
    text = re.sub(r"\bdidn t\b", "didn't", text)
    text = re.sub(r"\bwon t\b", "won't", text)
    text = re.sub(r"\bcant\b", "can't", text)
    text = re.sub(r"\bwouldnt\b", "wouldn't", text)
    text = re.sub(r"\bshouldnt\b", "shouldn't", text)
    text = re.sub(r"\bwasnt\b", "wasn't", text)
    text = re.sub(r"\bwere nt\b", "weren't", text)
    text = re.sub(r"\bhas nt\b", "hasn't", text)
    text = re.sub(r"\bhavent\b", "haven't", text)
    text = re.sub(r"\bhad nt\b", "hadn't", text)
    text = re.sub(r"\bdoes nt\b", "doesn't", text)
    text = re.sub(r"\bcould nt\b", "couldn't", text)
    text = re.sub(r"\bshould nt\b", "shouldn't", text)
    text = re.sub(r"\bwill not\b", "won't", text)
    text = re.sub(r"\bdo not\b", "don't", text)
    text = re.sub(r"\bcan not\b", "can't", text)
    text = re.sub(r"\bwould not\b", "wouldn't", text)
    text = re.sub(r"\bshould not\b", "shouldn't", text)
    text = re.sub(r"\bhave not\b", "haven't", text)
    text = re.sub(r"\bhas not\b", "hasn't", text)
    text = re.sub(r"\bhad not\b", "hadn't", text)
    text = re.sub(r"\bdoes not\b", "doesn't", text)
    text = re.sub(r"\bcould not\b", "couldn't", text)
    text = re.sub(r"\bshould not\b", "shouldn't", text)
    text = re.sub(r"\bdint\b", "didn't", text)
    text = re.sub(r"\bdidnt\b", "didn't", text)
    text = re.sub(r"\bwont\b", "won't", text)
    text = re.sub(r"\bdoesnt\b", "doesn't", text)
    text = re.sub(r"\bshouldnt\b", "shouldn't", text)
    text = re.sub(r"\bwasnt\b", "wasn't", text)
    text = re.sub(r"\bwerent\b", "weren't", text)
    text = re.sub(r"\bhavent\b", "haven't", text)
    text = re.sub(r"\bhadnt\b", "hadn't", text)
    text = re.sub(r"\bhasnt\b", "hasn't", text)
    # удаляем одиночные апострофы, не входящие в don't и т.п.
    text = re.sub(r"(?<!n)'(?!t\b)", " ", text)
    text = re.sub(r"'{2,}", " ", text)
    # лишние пробелы
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def prepare_data(file_path, sample_ratio=0.2, min_len=4, max_len=16, seq_len=10, random_seed=42):
    # основная функция подготовки данных
    random.seed(random_seed)
    np.random.seed(random_seed)

    # загрузка
    texts = load_tweets_from_text(file_path)
    print(f'всего строк в файле: {len(texts)}')

    # случайный сэмпл
    random.shuffle(texts)
    sample_size = int(len(texts) * sample_ratio)
    texts = texts[:sample_size]
    print(f'оставлено после сэмплирования: {len(texts)} строк')

    # очистка
    cleaned_texts = [clean_tweet(t) for t in texts]
    print('\nпримеры очистки:')
    for i in range(min(5, len(texts))):
        print(f'до:  {texts[i]}')
        print(f'после: {cleaned_texts[i]}\n')

    # фильтрация по длине
    filtered_texts = []
    for t in cleaned_texts:
        words = t.split()
        if min_len <= len(words) <= max_len:
            filtered_texts.append(t)
    print(f'после фильтрации по длине [{min_len}–{max_len}]: {len(filtered_texts)} текстов')

    # статистика по длине
    lengths = [len(t.split()) for t in filtered_texts]
    if lengths:
        print(f'средняя длина: {np.mean(lengths):.1f} слов')
        print(f'мин: {min(lengths)}, макс: {max(lengths)}')

    # разбивка на train/val/test
    train_texts, temp_texts = train_test_split(
        filtered_texts, test_size=0.2, random_state=random_seed, shuffle=True
    )
    val_texts, test_texts = train_test_split(
        temp_texts, test_size=0.5, random_state=random_seed, shuffle=True
    )
    print(f'train: {len(train_texts)}')
    print(f'val:   {len(val_texts)}')
    print(f'test:  {len(test_texts)}')

    # построение словаря только по train
    all_tokens = [word for text in train_texts for word in text.split()]
    vocab = ['<PAD>', '<UNK>'] + list(set(all_tokens))
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    vocab_size = len(vocab)

    print(f'размер словаря: {vocab_size}')
    print(f"примеры: 'love' -> {word_to_idx.get('love', 'не найдено')}, 'the' -> {word_to_idx.get('the', 'не найдено')}")

    return train_texts, val_texts, test_texts, word_to_idx, idx_to_word, vocab_size, seq_len


def create_data_loaders(train_texts, val_texts, test_texts, word_to_idx, seq_len, batch_size=128):
    # создаём датасеты и даталоадеры
    train_dataset = AutoregressiveTokenDataset(train_texts, word_to_idx, seq_len)
    val_dataset = AutoregressiveTokenDataset(val_texts, word_to_idx, seq_len)
    test_dataset = AutoregressiveTokenDataset(test_texts, word_to_idx, seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    print(f'размер train_dataset: {len(train_dataset)} пар (контекст → следующий токен)')
    print(f'пример из train_dataset[0]: input={train_dataset[0][0]}, target={train_dataset[0][1]}')

    # проверка батча
    x_batch, y_batch = next(iter(train_loader))
    print(f'x_batch.shape: {x_batch.shape}  # [b, seq_len]')
    print(f'y_batch.shape: {y_batch.shape}  # [b]')

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset


def save_artifacts(word_to_idx, idx_to_word, vocab_size, seq_len, train_dataset, val_dataset, test_dataset):
    # сохраняем словарь и датасеты
    with open('data/vocab_test.pkl', 'wb') as f:
        pickle.dump({
            'word_to_idx': word_to_idx,
            'idx_to_word': idx_to_word,
            'vocab_size': vocab_size,
            'seq_len': seq_len
        }, f)
    torch.save(train_dataset, 'data/train_dataset_test.pt')
    torch.save(val_dataset, 'data/val_dataset_test.pt')
    torch.save(test_dataset, 'data/test_dataset_test.pt')
    print('датасет готов, словарь сохранён')