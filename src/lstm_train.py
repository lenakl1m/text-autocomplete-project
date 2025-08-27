import os
import yaml
import torch
from tqdm import tqdm
from src.data_utils import tokenize_text
from src.eval_lstm import evaluate_model
from src.experiment_tracker import ExperimentTracker
from src.utils import generate_model_name
from src.utils import create_decode_fn



def train_epoch(model, dataloader, optimizer, criterion, device):
    # model: текущая модель
    # dataloader: загрузчик тренировочных данных
    # optimizer: оптимизатор
    # criterion: функция потерь 
    # device: cuda/cpu

    model.train()

    total_loss = 0.0
    total_tokens = 0

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
        num_tokens = (targets != 0).sum().item()
        total_tokens += num_tokens
        total_loss += loss.item()

    # возвращаю средний лосс
    return total_loss / total_tokens if total_tokens > 0 else float('inf')


def train_loop(model, train_loader, val_loader, test_loader, device, vocab, cfg, tracker=None):
    model_path = None
    torch.manual_seed(42)
    if device == 'cuda':
        torch.cuda.manual_seed_all(42)

    num_epochs = cfg['training']['num_epochs']

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='sum')

    decode_fn = create_decode_fn(vocab)

    # для ранней остановки
    best_val_loss = float('inf')
    patience = 5
    wait = 0

    os.makedirs("models", exist_ok=True)

    for epoch in range(num_epochs):
        # обучение
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        val_loss, val_rouge1, val_rouge2, val_examples = evaluate_model(
        model, val_loader, device, decode_fn, cfg
        )
        # метрики во время обучения
        print(f"[Epoch {epoch+1}/{num_epochs}] "
          f"Train: {train_loss:.3f} | "
          f"Val: {val_loss:.3f} | "
          f"ROUGE-1: {val_rouge1:.3f} | "
          f"ROUGE-2: {val_rouge2:.3f}")
        
        print("\nПримеры автодополнения (валидация):")
        for i, ex in enumerate(val_examples[:2]):
            print(f"  [{i+1}] Prefix: {ex['prefix']}")
            print(f"       True:  {ex['target']}")
            print(f"       Gen:   {ex['predicted']}")
            print()

        # сохраняем лучшую по val_loss модель
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_name = generate_model_name(cfg, model_type="lstm", extension="pth")
            model_path = os.path.join("models", model_name)

            torch.save(model.state_dict(), model_path)
            print(f"Лучшая модель сохранена: {model_name}")

            if tracker is not None:
                lstm_metrics = {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "test_loss": None,
                    "rouge1": val_rouge1,
                    "rouge2": val_rouge2,
                    "notes": f"Лучшая модель, epoch {epoch+1}"
                }
                tracker.log_lstm_experiment(
                    cfg=cfg,
                    metrics=lstm_metrics,
                    model_path=model_path
                )

            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Ранняя остановка - нет улучшений")
                break
        print("=" * 40)

    # считаем метрики на тесте
    test_loss, test_rouge1, test_rouge2, test_examples = evaluate_model(
        model, test_loader, device, decode_fn, cfg
    )

    print("\n" + "="*40)
    print("ОЦЕНКА НА ТЕСТЕ")
    print("="*40)
    print(f"Test Loss:     {test_loss:.3f}")
    print(f"ROUGE-1:       {test_rouge1:.3f}")
    print(f"ROUGE-2:       {test_rouge2:.3f}")

    if tracker is not None:
        final_metrics = {
            "train_loss": train_loss,
            "val_loss": best_val_loss,
            "test_loss": test_loss,
            "rouge1": test_rouge1,
            "rouge2": test_rouge2,
            "notes": "Финальная оценка на тесте"
        }
        tracker.log_lstm_experiment(
            cfg=cfg,
            metrics=final_metrics,
            model_path=model_path  # можно и без model_path, если не сохраняли
        )

    # примеры
    print("\nПримеры (тест):")
    for ex in test_examples[:3]:
        print(f"Prefix: {ex['prefix']}")
        print(f"True:  {ex['target']}")
        print(f"Gen:   {ex['predicted']}")
        print()

    return test_rouge1, test_rouge2