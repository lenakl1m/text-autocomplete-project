import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout,
             max_generation_length=20, temperature=1.0):
        # lstm-модель для предсказания следующего токена
        # vocab_size: размер словаря
        # embed_dim: размер эмбеддингов слов
        # hidden_dim: размер скрытого состояния LSTM
        # num_layers: количество слоёв LSTM
        # dropout: dropout для стабильности
        # max_generation_length: макс длина генерируемого текста
        # temperature: температура 

        super(LSTMModel, self).__init__()
        
        # слой эмбеддингов
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # lstm
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # выходной слой
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        # сохраняем параметры
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_generation_length = max_generation_length
        self.temperature = temperature
        
    def forward(self, x, hidden=None):
        # x текущий токен
        # hidden предыдущее скрытое состояние
        # возвращаем логиты для последнего токена, новое hidden

        embedded = self.embedding(x)  # (B, T) -> (B, T, E)
        lstm_out, hidden = self.lstm(embedded, hidden)
        logits = self.fc(lstm_out)

        return logits, hidden
    
    # автодополнение текста 
    def generate(self, start_tokens, method, temperature=None, forbidden_tokens=None, **kwargs):
        temp = temperature if temperature is not None else self.temperature
        self.eval() 
        device = next(self.parameters()).device

        # подготовка входа
        if isinstance(start_tokens, list):
            input_seq = torch.tensor(start_tokens, dtype=torch.long).unsqueeze(0).to(device)
            generated = start_tokens[:]
        else:
            input_seq = start_tokens.unsqueeze(0).to(device)
            generated = start_tokens.tolist()
        
        with torch.no_grad():
            embedded = self.embedding(input_seq)
            _, (h, c) = self.lstm(embedded)

        prefix_len = len(generated)
        if method == 'by_max_length':
            max_steps = kwargs.get('max_length', self.max_generation_length)
        elif method == 'by_num_words':
            max_steps = kwargs.get('num_words', 1)
        elif method == 'by_quarter_rule':
            # длина дополнения - 1/3 длины префикса 
            max_steps = max(5, prefix_len // 3)
        else:
            raise ValueError("method must be 'by_max_length', 'by_num_words', or 'by_quarter_rule'")
        print(f"Method: {method}, max_steps: {max_steps}")
        for _ in range(max_steps):
        
            # только последний токен
            input_token = input_seq[:, -1:]  # (1, 1)
                
            logits, (h, c) = self(input_token, (h, c)) 
            logits = logits.squeeze() / temp
            probs = torch.softmax(logits, dim=-1)

            # Запрещённые токены
            if forbidden_tokens is not None:
                probs[forbidden_tokens] = 0
            else:
                probs[0] = 0  # padding
                probs[1] = 0  # unk 

            if not torch.isfinite(probs).all():
                print("probs содержит nan/inf, пропускаем")
                continue
            if probs.sum() <= 0:
                probs = torch.ones_like(probs) / len(probs)
            else:
                probs /= probs.sum()

            next_token_idx = torch.multinomial(probs, num_samples=1).item()

            # Остановка по EOS (если используешь)
            # if next_token_idx == 2:
            #     generated.append(next_token_idx)
            #     break  # выход из цикла
            generated.append(next_token_idx)
            new_token = torch.tensor([[next_token_idx]], device=device)
            input_seq = torch.cat([input_seq, new_token], dim=1)

        return generated