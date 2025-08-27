import pandas as pd
from datetime import datetime
from pathlib import Path


class ExperimentTracker:
    def __init__(self, log_dir="data"):
        self.log_dir = Path(log_dir)
        self.csv_path = self.log_dir / "experiments.csv"
        self._init_csv() 

    def _init_csv(self):
        # пустой csv с нужными колонками
        columns = [
            "timestamp", "model_type",
            "vocab_size", "min_freq", "batch_size",
            "embed_dim", "hidden_dim", "num_layers",
            "max_generation_length", "temperature",
            "generation_method", "num_words", "max_length",
            "train_loss", "val_loss", "test_loss",
            "rouge1_lstm", "rouge2_lstm",
            "rouge1_distilgpt2", "rouge2_distilgpt2",
            "notes", "model_path"
        ]
        # файла нет - создаём пустой
        if not self.csv_path.exists():
            df = pd.DataFrame(columns=columns)
            df.to_csv(self.csv_path, index=False)

    def log_lstm_experiment(self, cfg, metrics, model_path=None):
        # логи lstm
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # строка для добавления в таблицу
        row = {
            "timestamp": timestamp,
            "model_type": "lstm",
            "vocab_size": cfg['training']['max_vocab_size'],
            "min_freq": cfg['training']['min_freq'],
            "batch_size": cfg['training']['batch_size'],
            "embed_dim": cfg['model']['embed_dim'],
            "hidden_dim": cfg['model']['hidden_dim'],
            "num_layers": cfg['model']['num_layers'],
            "max_generation_length": cfg['model']['max_generation_length'],
            "temperature": cfg['model']['temperature'],
            "generation_method": cfg['generation']['method'],
            "num_words": cfg['generation'].get('num_words', ''),
            "max_length": cfg['generation'].get('max_length', ''),
            "train_loss": metrics.get('train_loss', ''),
            "val_loss": metrics.get('val_loss', ''),
            "test_loss": metrics.get('test_loss', ''),
            "rouge1_lstm": metrics.get('rouge1', ''),
            "rouge2_lstm": metrics.get('rouge2', ''),
            "rouge1_distilgpt2": '',
            "rouge2_distilgpt2": '',
            "notes": metrics.get('notes', ''),
            "model_path": model_path,
        }

        # читаем текущую таблицу и добавляем новую строку
        df = pd.read_csv(self.csv_path)
        
        row_clean = {k: [v] for k, v in row.items()}
        new_row_df = pd.DataFrame(row_clean)

        if df.empty:
            df = new_row_df
        else:
            df = pd.concat([df, new_row_df], ignore_index=True)
        
        df.to_csv(self.csv_path, index=False)
        print(f"Эксперимент с LSTM добавлен")

    def log_transformer_experiment(self, cfg, metrics, model_name="distilgpt2"):
        # логи distilgpt2
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        row = {
            "timestamp": timestamp,
            "model_type": model_name,
            "vocab_size": '',
            "min_freq": '',
            "batch_size": '',
            "embed_dim": '',
            "hidden_dim": '',
            "num_layers": '',
            "max_generation_length": '',
            "temperature": cfg['transformer']['temperature'],
            "generation_method": '',
            "num_words": '',
            "max_length": '',
            "train_loss": '',
            "val_loss": '',
            "test_loss": '',
            "rouge1_lstm": '',
            "rouge2_lstm": '',
            "rouge1_distilgpt2": metrics.get('rouge1', ''),
            "rouge2_distilgpt2": metrics.get('rouge2', ''),
            "notes": metrics.get('notes', ''),
            "model_path": '',
            "config_path": ''
        }

        df = pd.read_csv(self.csv_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(self.csv_path, index=False)
        print(f"Эксперимент с {model_name} добавлен")

    def get_best_experiments(self, metric="rouge1_lstm", top_k=5):
        # возвращает топ-K лучших экспериментов по указанной метрике
        df = pd.read_csv(self.csv_path)
        df = df.dropna(subset=[metric])  # убираем, где метрика не задана
        df = df.sort_values(by=metric, ascending=False)
        return df.head(top_k)

    def compare_models(self):
        # показывает средние метрики lstm и distilgpt2
        df = pd.read_csv(self.csv_path)
        lstm = df[df['model_type'] == 'lstm']
        gpt = df[df['model_type'] == 'distilgpt2']

        print("Сравнение моделей по средним метрикам:")
        if not lstm.empty:
            mean_r1_lstm = lstm['rouge1_lstm'].mean()
            mean_r2_lstm = lstm['rouge2_lstm'].mean()
            print(f"LSTM       | ROUGE-1: {mean_r1_lstm:.4f}, ROUGE-2: {mean_r2_lstm:.4f}")
        if not gpt.empty:
            mean_r1_gpt = gpt['rouge1_distilgpt2'].mean()
            mean_r2_gpt = gpt['rouge2_distilgpt2'].mean()
            print(f"distilgpt2 | ROUGE-1: {mean_r1_gpt:.4f}, ROUGE-2: {mean_r2_gpt:.4f}")

    def show_recent_experiments(self, n=10):
        # показывает последние n экспериментов
        df = pd.read_csv(self.csv_path)
        print(f"\nПоследние {n} экспериментов:")
        print(df.tail(n).to_string(index=False))

    def get_model_comparison_df(tracker):
        df = pd.read_csv(tracker.csv_path)
        
        lstm = df[df['model_type'] == 'lstm']
        gpt = df[df['model_type'] == 'distilgpt2']
        
        data = []
        
        if not lstm.empty:
            data.append({
                'Model': 'LSTM',
                'ROUGE-1': lstm['rouge1_lstm'].mean(),
                'ROUGE-2': lstm['rouge2_lstm'].mean(),
                'Test Loss': lstm['test_loss'].mean(),
                'Experiments': len(lstm)
            })
        
        if not gpt.empty:
            data.append({
                'Model': 'DistilGPT2',
                'ROUGE-1': gpt['rouge1_distilgpt2'].mean(),
                'ROUGE-2': gpt['rouge2_distilgpt2'].mean(),
                'Test Loss': gpt['test_loss'].mean(),
                'Experiments': len(gpt)
            })
        
        return pd.DataFrame(data).round(4)
