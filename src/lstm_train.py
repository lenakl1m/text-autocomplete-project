import torch
from tqdm import tqdm
from src.data_utils import tokenize_text
from src.eval_lstm import evaluate_model


def train_epoch(model, dataloader, optimizer, criterion, device):
    # обучает модель один раз по всему тренировочному датасету
    # model текущая модель
    # dataloader загрузчик тренировочных данных
    # optimizer оптимизатор
    # criterion функция потерь (например, CrossEntropyLoss)
    # device cuda/cpu
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

def generate_fn(model, prompt, max_len, device, vocab, idx_to_word, k=5):
    # генерирует продолжение текста
    # model обученная модель
    # prompt строка — начало текста
    # max_len макс. число токенов для генерации
    # device cuda/cpu
    # vocab словарь word -> idx
    # idx_to_word: словарь idx -> word
    # k ширина beam (beam_width)

    model.eval()
    tokens = tokenize_text(prompt)
    indices = [vocab.get(t, vocab['<unk>']) for t in tokens]

    if not indices:
        return ""

    unk_idx = vocab['<unk>']
    pad_idx = vocab['<pad>']
    eos_idx = vocab.get('<eos>', 0)

    beam = [(indices.copy(), 0.0)]  # последовательность

    for step in range(max_len):
        all_candidates = []

        for seq, score in beam:
            input_tensor = torch.tensor([seq]).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                logits = output[0, -1].clone()

                # запрещаем <unk> и <pad>
                logits[unk_idx] = float('-inf')
                logits[pad_idx] = float('-inf')

                log_probs = torch.log_softmax(logits, dim=-1)
                top_log_probs, top_indices = torch.topk(log_probs, k)

                for i in range(k):
                    log_prob = top_log_probs[i].item()
                    token_id = top_indices[i].item()
                    new_seq = seq + [token_id]
                    new_score = score + log_prob
                    all_candidates.append((new_seq, new_score))

        # по убыванию вероятности
        all_candidates.sort(key=lambda x: x[1], reverse=True)
        # all_candidates.sort(key=lambda x: x[1] / len(x[0]), reverse=True)
        # оставляем топ k значений
        beam = all_candidates[:k]

        if all(seq[-1] == eos_idx or seq[-1] == 0 for seq, _ in beam):
            break

    # выбираем лучшую гипотезу
    best_seq = beam[0][0]
    # вырезаем продолжение (без префикса)
    gen_only = best_seq[len(indices):]

    # декодируем в текст
    generated_text = ' '.join([idx_to_word.get(idx, '<unk>') for idx in gen_only if idx != 0])
    return generated_text


def train_loop(model, train_loader, val_loader, device, vocab, num_epochs, k):
    # полный цикл обучения с валидацией, ранней остановкой и сохранением лучшей модели
    # model: обучаемая lstm-модель
    # train_loader: даталоадер для обучения
    # val_loader: даталоадер для валидации
    # device cuda/cpu
    # vocab словарь word -> idx
    # idx_to_word: словарь idx -> word
    # num_epochs: максимальное количество эпох

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

    # обратный словарь: idx -> word
    idx_to_word = {idx: word for word, idx in vocab.items()}

    def decode_fn(tokens):
        # преобразует индексы в слова
        return ' '.join([idx_to_word[t] for t in tokens if t != 0 and t in idx_to_word])

    best_val_loss = float('inf')
    patience = 4
    wait = 0

    for epoch in range(num_epochs):
        # обучение
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # валидация
        val_loss, val_rouge1, val_rouge2, examples = evaluate_model(
            model, val_loader, device, decode_fn, max_examples=100
        )

        print(f'epoch {epoch+1}, train loss: {train_loss:.4f}, val loss: {val_loss:.4f}')
        print(f'rouge-1: {val_rouge1:.4f}, rouge-2: {val_rouge2:.4f}')

        # примеры генерации
        print("\nПримеры автодополнения:")
        model.eval()
        with torch.no_grad():
            for inputs, targets in val_loader:
                chosen_indices = [8, 19]
                valid_indices = [i for i in chosen_indices if i < inputs.size(0)]
                invalid_indices = [i for i in chosen_indices if i >= inputs.size(0)]

                if invalid_indices:
                    print(f"Внимание: индексы {invalid_indices} выходят за пределы батча (размер {inputs.size(0)}). Пропущены.")

                for i in valid_indices:
                    # вход 
                    prefix_indices = inputs[i].cpu().numpy()
                    prefix_words = [idx_to_word[idx] for idx in prefix_indices if idx != 0 and idx in idx_to_word]
                    prefix = ' '.join(prefix_words)

                   # цель
                    target_indices = targets[i].cpu().numpy()
                    true_continuation_words = [idx_to_word[idx] for idx in target_indices if idx != 0 and idx in idx_to_word]
                    true_continuation = ' '.join(true_continuation_words)

                    # генерация
                    generated_continuation = generate_fn(
                        model,
                        prefix,
                        max_len=20,
                        device=device,
                        vocab=vocab,
                        idx_to_word=idx_to_word,
                        k=10  # или beam_width, зависит от реализации
                    )

                    # оставляем только продолжение
                    # если generate_fn возвращает полный текст (prefix + gen), тогда
                    if generated_continuation.startswith(prefix):
                        generated_only = generated_continuation[len(prefix):].strip()
                    else:
                        # если не начинается с префикса, берём как есть
                        generated_only = generated_continuation

                    # Выводим
                    print(f"prefix: {prefix}")
                    print(f"true:  {true_continuation}")
                    print(f"gen:   {generated_only}")
                    print("-" * 50)

                break

        # сохранение лучшей модели
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'models/lstm_model.pth')
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("ранняя остановка")
                break