import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import pickle

from src.lstm_model import LSTMTokenizerModel, generate
from src.eval_lstm import evaluate_with_rouge

# глобальные переменные для early stopping
best_val_loss = float('inf')
early_stopping_patience = 7
lr_scheduler_patience = 3
wait = 0

def load_data():
    # загрузка словаря
    with open('data/vocab_final.pkl', 'rb') as f:
        vocab_data = pickle.load(f)

    word_to_idx = vocab_data['word_to_idx']
    idx_to_word = vocab_data['idx_to_word']
    vocab_size = vocab_data['vocab_size']
    seq_len = vocab_data['seq_len']

    print(f"словарь загружен: {vocab_size} токенов")
    print(f"пример: 'the' -> {word_to_idx.get('the', 'не найдено')}")
    print(f"длина контекста (seq_len): {seq_len}")

    # загрузка датасетов
    train_dataset = torch.load('data/train_dataset_final.pt')
    val_dataset = torch.load('data/val_dataset_final.pt')
    test_dataset = torch.load('data/test_dataset_final.pt')

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128)
    test_loader = DataLoader(test_dataset, batch_size=128)

    print(f"размер обучающей выборки: {len(train_dataset)}")
    print(f"пример входа: {train_dataset[0]} -> {idx_to_word[train_dataset[0][1].item()]}")

    return word_to_idx, idx_to_word, vocab_size, seq_len, train_loader, val_loader, test_loader


def train():
    global best_val_loss, wait

    # загрузка данных
    word_to_idx, idx_to_word, vocab_size, seq_len, train_loader, val_loader, test_loader = load_data()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"используем устройство: {device}")

    # создание модели
    model = LSTMTokenizerModel(vocab_size, embed_dim=128, hidden_dim=256, num_layers=2).to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=lr_scheduler_patience, min_lr=1e-7)

    # проверка данных
    print("-" * 40)
    print('\nдебаг: проверка данных')
    print("-" * 40)
    x_batch, y_batch = next(iter(train_loader))
    print("пример x_batch[0]:", x_batch[0].tolist())
    print("y_batch.min():", y_batch.min().item())
    print("y_batch.max():", y_batch.max().item())

    assert y_batch.min() >= 0, 'отрицательные метки!'
    assert y_batch.max() < vocab_size, f'метка {y_batch.max().item()} >= vocab_size {vocab_size}'

    # проверка forward
    model.eval()
    with torch.no_grad():
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
    print("forward и loss работают")

    # фиксированный пример для сравнения
    val_iter = iter(val_loader)
    fixed_batch = next(val_iter)
    fixed_context = fixed_batch[0][0]
    fixed_target = fixed_batch[1][0].item()

    context_words = [idx_to_word.get(idx.item(), '<UNK>') for idx in fixed_context if idx != word_to_idx['<PAD>']]
    fixed_context_str = ' '.join(context_words)
    fixed_true_word = idx_to_word.get(fixed_target, '<UNK>')

    print("-" * 40)
    print("фиксированный пример для сравнения каждую эпоху")
    print("-" * 40)
    print(f"контекст: {fixed_context_str}")
    print(f"реальное продолжение: {fixed_true_word}")
    print(f"ожидаем: {fixed_context_str} {fixed_true_word}")
    print("-" * 40)

    # обучение
    n_epochs = 50
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in tqdm(train_loader, desc=f'epoch {epoch+1}'):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # валидация
        val_loss, val_acc, val_ppl, val_rouge = evaluate_with_rouge(model, val_loader, word_to_idx, idx_to_word, seq_len, device)

        print(f"epoch {epoch+1} | "
              f"train loss: {train_loss:.3f} | "
              f"val loss: {val_loss:.3f} | "
              f"val acc: {val_acc:.2%} | "
              f"val ppl: {val_ppl:.2f} | "
              f"val rouge-l: {val_rouge:.3f}")

        # сравнение с фиксированным примером
        print("-" * 40)
        print("прогресс модели (сравнение с истиной)")
        print("-" * 40)
        context_tensor = fixed_context.unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(context_tensor)
            pred_id = torch.argmax(logits, dim=1).item()
            probs = torch.softmax(logits, dim=1)
            top_probs, top_indices = torch.topk(probs, 3)

        pred_word = idx_to_word.get(pred_id, '<UNK>')
        top_words = [idx_to_word.get(idx.item(), '<UNK>') for idx in top_indices[0]]

        print(f"предсказание: {pred_word}")
        print(f"топ-3: {top_words}")
        print(f"реальность:   {fixed_true_word}")
        print("-" * 40)

        # сохранение лучшей модели
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
            torch.save(model.state_dict(), 'models/best_lstm.pt')
        else:
            wait += 1
            if wait >= early_stopping_patience:
                print(f"ранняя остановка на эпохе {epoch+1}")
                break

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"epoch {epoch+1} | lr: {current_lr:.2e} | val loss: {val_loss:.3f}")

    # загрузка лучшей модели
    model.load_state_dict(torch.load('best_lstm_model_final.pt'))
    print("загружена лучшая модель")

    return model, word_to_idx, idx_to_word, seq_len, device, test_loader