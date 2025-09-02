import torch
from torch.utils.data import Dataset

class AutoregressiveTokenDataset(Dataset):
    # датасет для предсказания следующего слова по контексту
    def __init__(self, texts, word_to_idx, seq_len):
        self.input_ids = []
        self.target_ids = []
        self.seq_len = seq_len
        self.word_to_idx = word_to_idx

        for text in texts:
            tokens = text.split()
            ids = [word_to_idx.get(t, word_to_idx['<UNK>']) for t in tokens]

            for i in range(1, len(ids)):
                end_idx = i
                start_idx = max(0, i - seq_len)
                context = ids[start_idx:end_idx]

                if len(context) < seq_len:
                    pad_id = word_to_idx['<PAD>']
                    context = [pad_id] * (seq_len - len(context)) + context

                self.input_ids.append(context)
                self.target_ids.append(ids[i])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.input_ids[idx], dtype=torch.long),
            torch.tensor(self.target_ids[idx], dtype=torch.long)
        )