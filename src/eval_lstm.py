import torch
from rouge_score import rouge_scorer

def evaluate_model(model, dataloader, device, decode_fn, max_examples=100):
    # model: обученная модель
    # dataloader: загрузчик данных (val/test), даёт батчи (input, target)
    # device: cuda/cpu
    # decode_fn: переводит индексы токенов в текст
    # max_examples: сколько примеров максимум обработать (для скорости)

    # перевожу модель в режим оценки
    model.eval()
    # инициализирую подсчёт rouge
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
    total_loss = 0.0
    total_rouge1 = 0.0
    total_rouge2 = 0.0
    count = 0
    # критерий — перекрёстная энтропия, игнорируем паддинг (0)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
    examples = []  # сохраняем пару примеров для анализа

    with torch.no_grad():
        for inputs, targets in dataloader:
            if count >= max_examples:
                break
            # переношу данные на устройство
            inputs, targets = inputs.to(device), targets.to(device)
            # прямой проход
            outputs, _ = model(inputs)
            # считаю лосс
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            total_loss += loss.item()

            # беру индексы наиболее вероятных токенов
            preds = outputs.argmax(dim=-1)
            for i in range(inputs.size(0)):
                # декодирую предсказание и настоящий текст
                pred_text = decode_fn(preds[i].cpu().numpy())
                target_text = decode_fn(targets[i].cpu().numpy())
                # сравниваю через rouge
                scores = scorer.score(target_text, pred_text)
                total_rouge1 += scores['rouge1'].fmeasure
                total_rouge2 += scores['rouge2'].fmeasure
                # сохраняю первые 5 примеров
                if len(examples) < 5:
                    examples.append({'target': target_text, 'predicted': pred_text})
                count += 1

    # усредняю метрики
    avg_loss = total_loss / (count * inputs.size(1))
    avg_rouge1 = total_rouge1 / count
    avg_rouge2 = total_rouge2 / count
    return avg_loss, avg_rouge1, avg_rouge2, examples