import torch
from tqdm import tqdm
from src.data_utils import tokenize_text
from src.eval_lstm import evaluate_model


def train_epoch(model, dataloader, optimizer, criterion, device):
    # model: текущая модель
    # dataloader: загрузчик тренировочных данных
    # optimizer: оптимизатор
    # criterion: функция потерь 
    # device: cuda/cpu

    model.train()

    total_loss = 0.0

    for inputs, targets in tqdm(dataloader, desc="Training"):

        # переношу данные на gpu
        inputs, targets = inputs.to(device), targets.to(device)

        # обнуляю градиенты
        optimizer.zero_grad()

        # получаю выход модели, берём только logits
        outputs, _ = model(inputs) 

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


def train_loop(model, train_loader, val_loader, test_loader, device, vocab, num_epochs, k=5,
               generation_method='by_num_words',  # выбираем метод генерации
               eval_during_training=True,         # флаг замера метрик во время обучения
               **gen_kwargs):                     # num_words=1 или max_length=20
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

    # для ранней остановки
    best_val_loss = float('inf')
    patience = 7
    wait = 0

    # если метрики считаем после
    final_examples = [] 

    for epoch in range(num_epochs):
        # обучение
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # метрики во время обучения
        if eval_during_training:
            val_loss, val_rouge1, val_rouge2, examples = evaluate_model(
                model, val_loader, device, decode_fn, max_examples=100
            )
            print(f'Epoch  {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            print(f'ROUGE-1: {val_rouge1:.4f}, ROUGE-2: {val_rouge2:.4f}')

            # Покажем пару примеров генерации
            print("\nПримеры автодополнения:")
            model.eval()
            with torch.no_grad():
                for inputs, targets in val_loader:
                    # примеры с фиксированных позиций, если они есть в батче
                    chosen_indices = [7, 22]
                    valid_indices = [i for i in chosen_indices if i < inputs.size(0)]

                    for i in valid_indices:
                        # префикс
                        prefix_indices = inputs[i].cpu().numpy()
                        prefix_words = [idx_to_word[idx] for idx in prefix_indices if idx != 0 and idx in idx_to_word]
                        prefix = ' '.join(prefix_words)

                        # настоящее
                        target_indices = targets[i].cpu().numpy()
                        true_continuation = ' '.join([idx_to_word[idx] for idx in target_indices if idx != 0 and idx in idx_to_word])

                        # токенизируем префикс и переводим в индексы
                        tokenized = tokenize_text(prefix)
                        start_indices = [vocab.get(t, vocab['<unk>']) for t in tokenized if t in vocab]

                        if len(start_indices) == 0:
                            continue

                        # генерируем продолжение с выбранным методом
                        generated_indices = model.generate(
                            start_tokens=start_indices,
                            method=generation_method,
                            **gen_kwargs  # num_words или max_length
                        )

                        # оставляем только сгенерированную часть
                        gen_only = generated_indices[len(start_indices):]
                        generated_text = ' '.join([idx_to_word.get(idx, '<unk>') for idx in gen_only if idx != 0])

                        # выводим для сравнения
                        print(f"Prefix: {prefix}")
                        print(f"True: {true_continuation}")
                        print(f"Gen:  {generated_text}")
                        print("-" * 40)
                    break
        else:
            val_loss = None
            print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}')

        # сохраняем лучшую по val_loss модель
        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'models/lstm_model.pth')
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Ранняя остановка - нет улучшений")
                break

    # в конце обучения
    print("\n" + "="*50)
    print("Оценка лучшей модели на тесте")
    print("="*50)

    # загружаем лучшую модель
    model.load_state_dict(torch.load('models/lstm_model.pth', weights_only=True))
    model.eval()

    # считаем метрики на тесте
    test_loss, test_rouge1, test_rouge2, test_examples = evaluate_model(
        model, test_loader, device, decode_fn, max_examples=500
    )

    print(f'Test loss: {test_loss:.4f}, ROUGE-1: {test_rouge1:.4f}, ROUGE-2: {test_rouge2:.4f}')

    # примеры
    print("\nПримеры с теста:")
    for ex in test_examples[:3]:
        print(f"True:  {ex['target']}")
        print(f"Gen: {ex['predicted']}")
        print("-" * 40)

    return test_rouge1, test_rouge2