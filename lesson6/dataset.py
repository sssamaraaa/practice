import os
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer


class TextDataset(Dataset):
    def __init__(self, text_path, tokenizer_path, max_length=128):
        self.max_length = max_length
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.add_special_tokens(["<pad>", "<s>", "</s>"])
        self.pad_token_id = self.tokenizer.token_to_id("<pad>")
        self.bos_token_id = self.tokenizer.token_to_id("<s>")
        self.eos_token_id = self.tokenizer.token_to_id("</s>")

        with open(text_path, "r", encoding="utf8") as f:
            text = f.read()

        # разбиваем книгу на абзацы
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        self.samples = []

        for paragraph in paragraphs:
            ids = self.tokenizer.encode(paragraph).ids
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
            # длинные абзацы режем окнами
            start = 0
            while start < len(ids):
                chunk = ids[start:start + max_length]
                if len(chunk) >= 2:
                    self.samples.append(chunk)
                start += max_length // 2

        print(f"Loaded {len(self.samples)} samples")

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def get_pad_token_id(self):
        return self.pad_token_id

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = self.samples[idx]
        x = ids[:-1]
        y = ids[1:]
        
        return {
            "input_ids": torch.tensor(x, dtype=torch.long),
            "target_ids": torch.tensor(y, dtype=torch.long)
        }


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        xs = [item["input_ids"] for item in batch]
        ys = [item["target_ids"] for item in batch]
        xs = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=self.pad_id)
        ys = torch.nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=self.pad_id)

        return {
            "input_ids": xs,
            "target_ids": ys
        }


def create_dataloader(text_path="data/book.txt", tokenizer_path="lesson6\mistral_tokenizer.json", batch_size=1, max_length=128, shuffle=True):
    dataset = TextDataset(text_path=text_path, tokenizer_path=tokenizer_path, max_length=max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=Collator(dataset.get_pad_token_id()), pin_memory=True)

    return loader, dataset