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
        
    def forward(self, x, hidden=None):
        # x текущий токен
        # hidden предыдущее скрытое состояние
        # возвращаем логиты для последнего токена, новое hidden

        embedded = self.embedding(x)  # (B, T) -> (B, T, E)
        lstm_out, hidden = self.lstm(embedded, hidden)
        logits = self.fc(lstm_out)

        return logits, hidden
    
    # автодополнение текста 
    def generate(self, start_tokens, 
                 method='by_max_length', # выбор метода 
                 temperature=None, 
                 **kwargs): # num_words=1 или max_length=20
        
        temp = temperature if temperature is not None else self.temperature
        self.eval()

        with torch.no_grad():
            # подготовка входа
            if isinstance(start_tokens, list):
                input_seq = torch.tensor(start_tokens, dtype=torch.long).unsqueeze(0)  
            else:
                input_seq = start_tokens.unsqueeze(0)  # если уже тензор
            
            # инициализация скрытых состояний
            h = torch.zeros(self.num_layers, 1, self.hidden_dim).to(input_seq.device)
            c = torch.zeros(self.num_layers, 1, self.hidden_dim).to(input_seq.device)

            generated = start_tokens.copy() if isinstance(start_tokens, list) else start_tokens.tolist()
            
            prefix_len = len(generated)
            
            if method == 'by_max_length':
                max_steps = kwargs.get('max_length', self.max_generation_length)
            elif method == 'by_num_words':
                max_steps = kwargs.get('num_words', 1)
            elif method == 'by_quarter_rule':
                # хотим сгенерить 1/4 от общей длины -> total = 4/3 * prefix_len
                total_len = int(prefix_len * 4 / 3)
                max_steps = total_len - prefix_len
                max_steps = max(1, max_steps)
            else:
                raise ValueError("method must be 'by_max_length', 'by_num_words', or 'by_quarter_rule'")

            for _ in range(max_steps):
                # только последний токен
                input_token = input_seq[:, -1:]  # (1, 1)
                
                # эмбеддинг
                embedded = self.embedding(input_token)
                
                # lstm с сохранением состояния
                lstm_out, (h, c) = self.lstm(embedded, (h, c))
                
                # логиты
                logits = self.fc(lstm_out).squeeze() / temperature
                
                # применяем softmax
                probs = torch.softmax(logits, dim=-1)
                next_token_idx = torch.multinomial(probs, num_samples=1).item()
                
                # добавляем к результату
                generated.append(next_token_idx)
                
                # обновляем вход
                input_seq = torch.cat([input_seq, torch.tensor([[next_token_idx]]).to(input_seq.device)], dim=1)
        
        return generated