import torch
import torch.nn as nn

class LSTMTokenizerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        embed = self.embedding(x)
        lstm_out, _ = self.lstm(embed)
        logits = self.fc(lstm_out[:, -1, :])
        return logits


def generate(model, seed_words, word_to_idx, idx_to_word, seq_len, max_len=15, temperature=1.0, top_k=None):
    # перевести модель в режим оценки
    model.eval()
    words = seed_words.split()
    ids = [word_to_idx.get(w, word_to_idx['<UNK>']) for w in words[-seq_len:]]

    # дополнить паддингом, если короткий контекст
    if len(ids) < seq_len:
        ids = [word_to_idx['<PAD>']] * (seq_len - len(ids)) + ids
    else:
        ids = ids[-seq_len:]

    device = next(model.parameters()).device
    context = torch.tensor([ids], dtype=torch.long).to(device)
    output = words[:]

    for _ in range(max_len):
        with torch.no_grad():
            logits = model(context)
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)

            # top-k фильтрация
            if top_k is not None:
                top_probs, top_indices = torch.topk(probs, top_k)
                shifted_probs = torch.zeros_like(probs).scatter_(1, top_indices, top_probs)
                probs = shifted_probs / shifted_probs.sum()

            pred_id = torch.multinomial(probs, num_samples=1).item()

        pred_word = idx_to_word.get(pred_id, '<UNK>')

        # защита от повтора последнего слова
        if len(output) > 0 and pred_word == output[-1]:
            probs_sorted = torch.argsort(logits, descending=True)
            for idx in probs_sorted[0]:
                word = idx_to_word.get(idx.item(), '<UNK>')
                if word != output[-1]:
                    pred_id = idx.item()
                    pred_word = word
                    break

        output.append(pred_word)

        # сдвиг контекста
        new_ids = ids[1:] + [pred_id]
        context = torch.tensor([new_ids], dtype=torch.long).to(device)
        ids = new_ids

        # остановка на знаке препинания или длине
        if pred_word in ['.', '!', '?'] or len(output) > 20:
            break

    return " ".join(output)