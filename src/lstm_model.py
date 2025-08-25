import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout):
        # lstm-модель для предсказания следующего токена
        # vocab_size: размер словаря
        # embed_dim: размер эмбеддингов слов
        # hidden_dim: размер скрытого состояния LSTM
        # num_layers: количество слоёв LSTM
        # dropout: dropout для стабильности

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
        
    def forward(self, x, hidden=None):
        # x текущий токен
        # hidden предыдущее скрытое состояние
        # возвращаем логиты для последнего токена, новое hidden

        embedded = self.embedding(x)  # (B, T) -> (B, T, E)
        lstm_out, hidden = self.lstm(embedded, hidden)  # передаём hidden
        logits = self.fc(lstm_out)

        return logits, hidden
    
    def generate(self, start_tokens, max_length=20, temperature=1.0):
        # автодополнение текста 
        # :param start_tokens: список индексов или тензор (seq_len,) — начало текста
        # :param max_length: максимальная длина генерации
        # :param temperature: температура для "случайности" (чем выше — тем случайнее)
        # :return: список индексов длины <= max_length

        self.eval()
        with torch.no_grad():
            # подготовка входа
            if isinstance(start_tokens, list):
                input_seq = torch.tensor(start_tokens, dtype=torch.long).unsqueeze(0)  # (1, seq_len)
            else:
                input_seq = start_tokens.unsqueeze(0)  # если уже тензор
            
            generated = start_tokens.copy() if isinstance(start_tokens, list) else start_tokens.tolist()
            
            # инициализация скрытых состояний
            h = torch.zeros(self.num_layers, 1, self.hidden_dim).to(input_seq.device)
            c = torch.zeros(self.num_layers, 1, self.hidden_dim).to(input_seq.device)
            
            for _ in range(max_length):
                # только последний токен
                input_token = input_seq[:, -1:]  # (1, 1)
                
                # эмбеддинг
                embedded = self.embedding(input_token)
                
                # lstm с сохранением состояния
                lstm_out, (h, c) = self.lstm(embedded, (h, c))
                
                # логиты
                logits = self.fc(lstm_out).squeeze() / temperature  # (vocab_size,)
                
                # применяем softmax
                probs = torch.softmax(logits, dim=-1)
                next_token_idx = torch.multinomial(probs, num_samples=1).item()
                
                # добавляем к результату
                generated.append(next_token_idx)
                
                # обновляем вход
                input_seq = torch.cat([input_seq, torch.tensor([[next_token_idx]]).to(input_seq.device)], dim=1)
        
        return generated