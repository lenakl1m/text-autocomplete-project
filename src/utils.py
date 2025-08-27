import os
import glob
import torch
from datetime import datetime
from src.lstm_model import LSTMModel

def generate_model_name(cfg, model_type="lstm", extension="pth"):
    # извлекаем параметры
    vocab_size = cfg['training']['max_vocab_size']
    batch_size = cfg['training']['batch_size']
    embed_dim = cfg['model']['embed_dim']
    hidden_dim = cfg['model']['hidden_dim']
    num_epochs = cfg['training']['num_epochs']
    
    # дата и время
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    # создаём имя
    name = (
        f"{model_type}"
        f"__vocab{vocab_size//1000}k"
        f"__bs{batch_size}"
        f"__emb{embed_dim}"
        f"__h{hidden_dim}"
        f"__ep{num_epochs}"
        f"__{timestamp}.{extension}"
    )
    return name

def load_best_model(cfg, vocab, device):
    # все pth файлы
    model_files = glob.glob("models/lstm__*.pth")
    if not model_files:
        raise FileNotFoundError("Нет сохранённых моделей")

    # самый свежий по времени
    latest_model = max(model_files, key=os.path.getctime)

    # модель с параметрами из конфига
    model = LSTMModel(
        vocab_size=len(vocab),
        embed_dim=cfg['model']['embed_dim'],
        hidden_dim=cfg['model']['hidden_dim'],
        num_layers=cfg['model']['num_layers'],
        dropout=cfg['model']['dropout'],
        max_generation_length=cfg['model']['max_generation_length'],
        temperature=cfg['model']['temperature']
    ).to(device)

    # загружаем веса
    model.load_state_dict(torch.load(latest_model, weights_only=True))
    model.eval()
    print(f"Загружена модель: {latest_model}")
    return model

def create_decode_fn(vocab):
    idx_to_word = {idx: word for word, idx in vocab.items()}
    def decode_fn(tokens):
        return ' '.join([idx_to_word[t] for t in tokens if t != 0 and t in idx_to_word])
    return decode_fn