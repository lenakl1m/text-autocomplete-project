import os
import pandas as pd
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from typing import List

from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from src.next_token_dataset import NextTokenDataset

# качаем ресурсы для nltk
nltk.download('punkt_tab', quiet=True)
# крайние меры
# nltk.download('book', quiet=True)  
# nltk.download('all', quiet=True)

# очистка (оставляем только слова и пробелы)
def clean_text(text: str) -> str:
    # нижний регистр
    text = text.lower()
    
    # упоминания (@username)
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    
    # ссылки 
    text = re.sub(r'https?://[A-Za-z0-9./]+', '', text)
    
    #эмодзи и не ascii символы
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    
    # пунктуациz 
    text = re.sub(r'[' + string.punctuation + ']', ' ', text)
    
    #лишние пробелы
    text = re.sub(r'\s+', ' ', text)
    
    #пробелы в начале/конце
    text = text.strip()
    
    return text

def tokenize_text(text: str) -> List[str]:
    # токенизируем текст с помощью nlkt
    return word_tokenize(text)

def load_and_clean_dataset(raw_path: str, processed_path: str):
    # проверяем, существует ли уже обработанный файл
    if os.path.exists(processed_path):
        print(f"{processed_path} уже существует, загружаю...")
        return pd.read_csv(processed_path)

    # загружаем сырой датасет, очищаем и сохраняем
    # построчно
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

   #cохраняем
    df.to_csv(processed_path, index=False)
    print(f"Обработанный датасет сохранён в {processed_path}")
    print(f"\nКоличество текстов: {len(df)}")
    return df


# разбивка на train (80%), val (10%), test (10%)

def split_dataset(df, train_path="data/train.csv", val_path="data/val.csv", test_path="data/test.csv"):
    train, temp = train_test_split(df, test_size=0.2, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)
    
    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)
    test.to_csv(test_path, index=False)
    
    print(f"Разделение:")
    print(f"\tТренировочный: {len(train)}")
    print(f"\tВалидационный: {len(val)}")
    print(f"\tТестовый: {len(test)}")

    return train, val, test

def collate_fn(batch):
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    # паддинг до максимальной длины в батче
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)
    return inputs_padded, targets_padded

def get_data_loaders(train_path="data/train.csv",
                     val_path="data/val.csv",
                     test_path="data/test.csv",
                     batch_size=64,
                     max_vocab_size=5000):

    # датасеты 
    train_dataset = NextTokenDataset(train_path, max_vocab_size=max_vocab_size)
    val_dataset = NextTokenDataset(val_path, vocab=train_dataset.get_vocab())
    test_dataset = NextTokenDataset(test_path, vocab=train_dataset.get_vocab())

    # даталоадеры
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader, train_dataset.get_vocab()