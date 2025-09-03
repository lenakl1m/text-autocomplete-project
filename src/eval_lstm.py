import torch
import numpy as np
from rouge_score import rouge_scorer
from src.lstm_model import generate

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def evaluate_with_rouge(model, loader, word_to_idx, idx_to_word, seq_len, device):
    # оценка модели: loss, acc, ppl, rouge-l
    model.eval()
    total_loss = 0.
    correct = 0
    total = 0
    rouge_scores = []
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

            # rouge-l по одному слову
            for i in range(len(y_batch)):
                true_word = idx_to_word.get(y_batch[i].item(), '<UNK>')
                pred_word = idx_to_word.get(preds[i].item(), '<UNK>')
                score = scorer.score(true_word, pred_word)
                rouge_scores.append(score['rougeL'].fmeasure)

    avg_loss = total_loss / len(loader)
    acc = correct / total
    perplexity = np.exp(avg_loss)
    avg_rouge_l = np.mean(rouge_scores) if rouge_scores else 0.0

    return avg_loss, acc, perplexity, avg_rouge_l


def evaluate_final(model, test_loader, word_to_idx, idx_to_word, seq_len, device):
    # финальная оценка на тесте + анализ генерации
    from collections import Counter

    # тестовые метрики
    test_loss, test_acc, test_ppl, test_rouge = evaluate_with_rouge(model, test_loader, word_to_idx, idx_to_word, seq_len, device)
    pprint("-" * 40)
    print("дополнительные метрики")
    print(f"test loss: {test_loss:.3f} | test acc: {test_acc:.2%} | test ppl: {test_ppl:.2f} | test rouge-l: {test_rouge:.3f}")
    print("-" * 40)

    # примеры генерации
    generated_examples = [
        generate(model, 'i', word_to_idx, idx_to_word, seq_len, temperature=0.8, top_k=10),
        generate(model, 'i love', word_to_idx, idx_to_word, seq_len, temperature=0.8, top_k=10),
        generate(model, 'today', word_to_idx, idx_to_word, seq_len, temperature=0.8, top_k=10),
        generate(model, 'life', word_to_idx, idx_to_word, seq_len, temperature=0.8, top_k=10),
        generate(model, 'i can\'t', word_to_idx, idx_to_word, seq_len, temperature=0.8, top_k=10),
    ]

    all_gen_words = []
    print("-" * 40)
    print("\nсгенерированные примеры")
    print("-" * 40)
    for gen in generated_examples:
        print('→', gen)
        all_gen_words.extend(gen.split())

    # статистика по генерации
    avg_len = np.mean([len(g.split()) for g in generated_examples])
    print(f"\nсредняя длина сгенерированной фразы: {avg_len:.1f} слов")

    unique_ratio = len(set(all_gen_words)) / len(all_gen_words) if all_gen_words else 0
    print(f"доля уникальных слов в генерации: {unique_ratio:.2%}")

    word_freq = Counter(all_gen_words)
    print(f"топ-5 слов в генерации: {word_freq.most_common(5)}")
    print("-" * 40)

    # количество параметров
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"количество обучаемых параметров: {n_params:,}")