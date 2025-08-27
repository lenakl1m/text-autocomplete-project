from transformers import AutoModelForCausalLM, AutoTokenizer
# from transformers import pipeline
from rouge_score import rouge_scorer
import torch
from tqdm import tqdm


def evaluate_transformer_pipeline(val_loader, cfg, device='cuda'):

    # оценка качества модели distilgpt2 на задаче автодополнения
    # val_loader: DataLoader с парами input/target, input — префикс (3/4), target — 1/4
    # max_examples: сколько примеров обработать 
    # device: cuda/cpu

    model_name = cfg['transformer']['model_name']
    max_examples = cfg['evaluation']['max_examples']
    max_returned = cfg['evaluation']['max_returned_examples']
    gen_config = cfg['transformer']

    # # pipeline
    # generator = pipeline(
    #     "text-generation",
    #     model=model_name,
    #     device=0 if device == 'cuda' else -1  # 0 = GPU, -1 = CPU
    # )
    
    if device == 'cuda' and torch.cuda.is_available():
        device_torch = torch.device('cuda')
    else:
        device_torch = torch.device('cpu')

    print(f"Используем устройство: {device_torch}")

    # токенизатор
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device_torch)
    model.eval()

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)

    total_rouge1 = 0.0
    total_rouge2 = 0.0
    count = 0
    examples = []

    # по валидационному датасету
    with torch.no_grad():
        for batch_inputs, batch_targets in tqdm(val_loader, desc="Evaluating distilgpt2"):
            # обрезаем до max_examples
            if count >= max_examples:
                break

            for i in range(batch_inputs.size(0)):
                if count >= max_examples:
                    break

                # декодируем input и target
                input_ids = batch_inputs[i].cpu().numpy()
                target_ids = batch_targets[i].cpu().numpy()

                # убираем паддинг
                input_ids = input_ids[input_ids != 0]
                target_ids = target_ids[target_ids != 0]

                # пропускаем пустые
                if len(input_ids) == 0 or len(target_ids) == 0:
                    continue

                # переводим в текст
                prefix = tokenizer.decode(input_ids, skip_special_tokens=True)
                true_continuation = tokenizer.decode(target_ids, skip_special_tokens=True)

                if not prefix.strip() or not true_continuation.strip():
                    continue

                # генерация
                try:
                    inputs = tokenizer(prefix, return_tensors="pt").to(device_torch)
                    num_new_tokens = len(target_ids)

                    current_length = len(input_ids)
                    target_length = len(target_ids)
                    max_length = current_length + target_length + 5  # запас

                    outputs = model.generate(
                        inputs.input_ids,
                        max_new_tokens=num_new_tokens,
                        do_sample=gen_config['do_sample'],
                        top_k=gen_config['top_k'],
                        temperature=gen_config['temperature'],
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )

                    generated_ids = outputs[0][inputs.input_ids.shape[-1]:]  # только новые
                    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

                    # rouge
                    if generated_text.strip():
                        scores = scorer.score(true_continuation, generated_text)
                        total_rouge1 += scores['rouge1'].fmeasure
                        total_rouge2 += scores['rouge2'].fmeasure
                    else:
                        total_rouge1 += 0.0
                        total_rouge2 += 0.0

                    # сохраняем примеры
                    if len(examples) < max_returned:
                        examples.append({
                            'prefix': prefix,
                            'true': true_continuation,
                            'generated': generated_text
                        })

                    count += 1

                except Exception as e:
                    print(f"Ошибка: {e}")
                    continue

    avg_rouge1 = total_rouge1 / count if count > 0 else 0
    avg_rouge2 = total_rouge2 / count if count > 0 else 0

    return avg_rouge1, avg_rouge2, examples