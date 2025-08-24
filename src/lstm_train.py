# src/lstm_train.py
import torch
from tqdm import tqdm
from src.data_utils import tokenize_text
from src.eval_lstm import evaluate_model


def train_epoch(model, dataloader, optimizer, criterion, device):
    # обучает модель один раз по всему тренировочному датасету
    # model: текущая модель (в режиме train)
    # dataloader: загрузчик тренировочных данных
    # optimizer: оптимизатор (например, Adam), обновляет веса
    # criterion: функция потерь (например, CrossEntropyLoss)
    # device: 'cuda' или 'cpu', куда переносить данные
    model.train()
    total_loss = 0.0
    for inputs, targets in tqdm(dataloader, desc="Training"):
        # переношу данные на gpu
        inputs, targets = inputs.to(device), targets.to(device)
        # обнуляю градиенты
        optimizer.zero_grad()
        # получаю выход модели
        outputs = model(inputs)
        # считаю лосс по всем токенам в батче
        loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        # обратное распространение
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # шаг оптимизатора
        optimizer.step()
        total_loss += loss.item()
    # возвращаю средний лосс
    return total_loss / len(dataloader)


def train_loop(model, train_loader, val_loader, device, vocab, num_epochs):
    # полный цикл обучения с валидацией, ранней остановкой и сохранением лучшей модели
    # model: обучаемая lstm-модель
    # train_loader: даталоадер для обучения
    # val_loader: даталоадер для валидации
    # device: устройство для вычислений
    # vocab: словарь (для декодирования индексов в слова)
    # tokenizer: функция токенизации текста (например, word_tokenize)
    # num_epochs: максимальное количество эпох

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

    idx_to_word = {idx: word for word, idx in vocab.items()}

    def decode_fn(tokens):
        # tokens - массив индексов
        return ' '.join([idx_to_word[t] for t in tokens if t != 0])

    def generate_fn(model, prompt, max_len, device, vocab, idx_to_word):
        # генерирует продолжение текста по заданному префиксу
        # model: обученная модель
        # prompt: строка — начало текста
        # max_len: макс. число токенов для генерации
        # device: где выполняется модель
        # idx_to_word: словарь для кодирования/декодирования
        model.eval()
        tokens = tokenize_text(prompt)
        indices = [vocab.get(t, vocab['<unk>']) for t in tokens]
        for _ in range(max_len):
            input_tensor = torch.tensor([indices]).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                pred_token = output[0, -1].argmax().item()
            if pred_token == 0 or pred_token == vocab['<pad>']:
                break
            indices.append(pred_token)
        return ' '.join([idx_to_word.get(idx, '<unk>') for idx in indices])

    best_val_loss = float('inf')
    patience = 4
    wait = 0

    for epoch in range(num_epochs):
        # обучение
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # валидация через eval_lstm
        val_loss, val_rouge1, val_rouge2, examples = evaluate_model(
            model, val_loader, device, decode_fn, max_examples=100
        )

        print(f'epoch {epoch+1}, train loss: {train_loss:.4f}, val loss: {val_loss:.4f}')
        print(f'rouge-1: {val_rouge1:.4f}, rouge-2: {val_rouge2:.4f}')

        # примеры автодополнения
        print("\nПримеры автодополнения:")
        for inputs, targets in val_loader:
            inputs = inputs[:2]  # первые два примера
            for i in range(inputs.size(0)):
                prefix_indices = inputs[i].cpu().numpy()
                # используем idx_to_word вместо vocab.lookup_token
                prefix_tokens = [idx_to_word[t] for t in prefix_indices if t != 0 and t in idx_to_word]
                prefix = ' '.join(prefix_tokens)

                generated = generate_fn(model, prefix, max_len=15, device=device, vocab=vocab, idx_to_word=idx_to_word)
                target = ' '.join([idx_to_word[t] for t in targets[i].cpu().numpy() if t != 0 and t in idx_to_word])

                print(f"prefix: {prefix}")
                print(f"true: {target}")
                print(f"gen:  {generated}")
                print("-" * 50)
            break

        # сохраняю лучшую модель
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'models/lstm_model.pth')
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("ранняя остановка")
                break