from transformers import AutoModelForCausalLM, AutoTokenizer
# from transformers import pipeline
from rouge_score import rouge_scorer
import torch
from tqdm import tqdm


def evaluate_transformer(val_texts, cfg, device='cuda'):

    # оценка качества модели distilgpt2 на задаче автодополнения
    # val_loader: DataLoader с парами input/target, input — префикс (3/4), target — 1/4
    # max_examples: сколько примеров обработать 
    # device: cuda/cpu

    model_name = cfg['transformer']['model_name']
    max_examples = cfg['evaluation']['max_examples']
    gen_config = cfg['transformer']

    # # pipeline
    # generator = pipeline(
    #     "text-generation",
    #     model=model_name,
    #     device=0 if device == 'cuda' else -1  # 0 = GPU, -1 = CPU
    # )
    
    device_torch = torch.device(device if torch.cuda.is_available() else 'cpu')

    print(f"Используем устройство: {device_torch}")

    # токенизатор
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device_torch)
    model.eval()

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=False)

    total_rouge1 = 0.0
    total_rouge2 = 0.0
    count = 0
    examples = []

    # по валидационному датасету
    with torch.no_grad():
        for text in tqdm(val_texts, desc="Evaluating distilgpt2", total=max_examples):
            if count >= max_examples:
                break

            text = text.strip()
            if not text:
                continue

            words = text.split()
            if len(words) < 8:
                continue

            split_idx = len(words) * 3 // 4
            if split_idx <= 1:
                continue

            prefix = " ".join(words[:split_idx])
            true_continuation = " ".join(words[split_idx:])

            if not prefix or not true_continuation:
                continue

            allowed_gen_keys = {
                "do_sample", "top_k", "top_p", "temperature", "repetition_penalty",
                "num_return_sequences", "no_repeat_ngram_size", "min_new_tokens"
            }

            gen_kwargs = {
                k: v for k, v in gen_config.items()
                if k in allowed_gen_keys
            }

            try:
                # токенизируем префикс
                inputs = tokenizer(prefix, return_tensors="pt", padding=False, truncation=True).to(device_torch)

                # сколько токенов генерировать
                true_token_count = tokenizer(
                    true_continuation,
                    add_special_tokens=False,
                    return_tensors="pt"
                )["input_ids"].shape[-1]
                max_new_tokens = min(max(5, true_token_count), 20)

                # генерация
                outputs = model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    **gen_kwargs
                )

                # декодируем только новые токены
                generated_ids = outputs[0, inputs.input_ids.size(-1):]
                generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

                # считаем rouge
                if generated_text.strip():
                    scores = scorer.score(true_continuation, generated_text)
                    total_rouge1 += scores['rouge1'].fmeasure
                    total_rouge2 += scores['rouge2'].fmeasure
                else:
                    total_rouge1 += 0.0
                    total_rouge2 += 0.0

                if len(examples) < 5:
                    examples.append({
                        'prefix': prefix,
                        'true': true_continuation,
                        'generated': generated_text
                    })
                count += 1

            except Exception as e:
                print(f"Ошибка при генерации: {e}")
                continue

    avg_rouge1 = total_rouge1 / count if count > 0 else 0.0
    avg_rouge2 = total_rouge2 / count if count > 0 else 0.0

    return avg_rouge1, avg_rouge2, examples