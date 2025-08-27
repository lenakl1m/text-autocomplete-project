import torch
from rouge_score import rouge_scorer


def evaluate_model(model, dataloader, device, decode_fn, cfg):
    # model: обученная модель
    # dataloader: загрузчик данных (val/test), даёт батчи (input, target)
    # device: cuda/cpu
    # decode_fn: переводит индексы токенов в текст
    # cfg: настройки

    # перевожу модель в режим оценки
    model.eval()

    # инициализирую подсчёт rouge
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=False)
    total_loss = 0.0
    total_rouge1 = 0.0
    total_rouge2 = 0.0
    count = 0
    total_tokens = 0
    # критерий — перекрёстная энтропия, игнорируем паддинг (0)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
    examples = []
    max_examples = cfg['evaluation']['max_examples']
    gen_method = cfg['generation']['method']
    gen_kwargs = {}

    if gen_method == 'by_num_words':
        gen_kwargs['num_words'] = cfg['generation']['num_words']
    elif gen_method == 'by_max_length':
        gen_kwargs['max_length'] = cfg['generation']['max_length']
    elif gen_method == 'by_quarter_rule':
        # ничего не нужно — `generate` сам посчитает
        pass
    else:
        raise ValueError(f"Unknown generation method: {gen_method}")

    with torch.no_grad():
        for inputs, targets in dataloader:
            if count >= max_examples:
                break
            # переношу данные на устройство
            inputs = inputs.to(device)
            targets = targets.to(device)

            # прямой проход
            outputs, _ = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            mask = (targets != 0)
            num_tokens = mask.sum().item()
            total_tokens += num_tokens
            total_loss += loss.item()

            for i in range(inputs.size(0)):
                if count >= max_examples:
                    break

                input_ids = inputs[i].cpu().numpy()
                if 0 in input_ids:
                    seq_len = (input_ids != 0).sum()
                    input_ids = input_ids[:seq_len]
                else:
                    seq_len = len(input_ids)

                if seq_len < 4:
                    continue

                prefix_len = seq_len * 3 // 4
                if prefix_len == 0:
                    continue

                prefix_ids = input_ids[:prefix_len]
                true_ids = input_ids[prefix_len:]

                generated_ids = model.generate(
                    start_tokens=prefix_ids.tolist(),
                    method=gen_method,
                    temperature=cfg['model']['temperature'],
                    **gen_kwargs  # используем собранный словарь
                )

                gen_only = generated_ids[len(prefix_ids):]

                if len(gen_only) == 0:
                    pred_text = ""
                else:
                    pred_text = decode_fn(gen_only)
                    
                true_text = decode_fn(true_ids)

                if not pred_text.strip() or not true_text.strip():
                    rouge1 = rouge2 = 0.0
                else:
                    scores = scorer.score(true_text, pred_text)
                    rouge1 = scores['rouge1'].fmeasure
                    rouge2 = scores['rouge2'].fmeasure

                total_rouge1 += rouge1
                total_rouge2 += rouge2

                if len(examples) < 5:
                    examples.append({
                        'prefix': decode_fn(prefix_ids),
                        'target': true_text,
                        'predicted': pred_text
                    })

                count += 1

    # усредняю метрики
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    avg_rouge1 = total_rouge1 / count if count > 0 else 0.0
    avg_rouge2 = total_rouge2 / count if count > 0 else 0.0

    return avg_loss, avg_rouge1, avg_rouge2, examples