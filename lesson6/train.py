import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.amp import autocast
from torch.amp import GradScaler
from dataset import create_dataloader
from transformer import GeneratorTransformer


class Trainer:
    def __init__(self, model, dataloader, dataset, config, device):
        self.model = model.to(device)
        self.loader = dataloader
        self.dataset = dataset
        self.device = device
        self.config = config
        self.criterion = nn.CrossEntropyLoss(ignore_index=dataset.get_pad_token_id())
        self.optimizer = optim.Adam(self.model.parameters(), lr=config["learning_rate"])
        #self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)
        self.scaler = GradScaler()
        self.save_dir = config["save_dir"]
        os.makedirs(self.save_dir, exist_ok=True)

    def save_checkpoint(self, epoch):
        path = os.path.join(self.save_dir, f"epoch_{epoch}.pt")

        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config
        }, path)

        print(f"Saved: {path}")

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        progress = tqdm(self.loader, desc=f"Epoch {epoch}")

        for i, batch in enumerate(progress):
            input_ids = batch["input_ids"].to(self.device)
            target_ids = batch["target_ids"].to(self.device)
            self.optimizer.zero_grad()

            with autocast(device_type="cuda", dtype=torch.float16):
                logits = self.model(input_ids)
                loss = self.criterion(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item()
            progress.set_postfix(loss=f"{total_loss/(i+1):.4f}")

        return total_loss / (i + 1)

    def train(self):
        epochs = self.config["num_epochs"]

        for epoch in range(1, epochs + 1):
            loss = self.train_epoch(epoch)
            print(f"Epoch {epoch}: loss={loss:.4f}")
            #self.scheduler.step()
            self.save_checkpoint(epoch)


def main():
    config = {
        "batch_size":32,
        "max_length": 128,
        "learning_rate": 3e-4,
        "num_epochs": 30,
        "save_dir": "lesson6/checkpoints",
        "d_model": 256,
        "num_heads": 8,
        "d_ff": 1024,
        "num_layers": 4,
        "dropout": 0.1
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader, dataset = create_dataloader(
        text_path="data/book.txt",
        tokenizer_path="lesson6\mistral_tokenizer.json",
        batch_size=config["batch_size"],
        max_length=config["max_length"]
    )

    model = GeneratorTransformer(
        vocab_size=dataset.get_vocab_size(),
        tokenizer=dataset.tokenizer,
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        num_layers=config["num_layers"],
        max_length=config["max_length"],
        pad_index=dataset.get_pad_token_id(),
        dropout=config["dropout"],
        device=device
    )

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    trainer = Trainer(model, loader, dataset, config, device)
    trainer.train()


if __name__ == "__main__":
    main()