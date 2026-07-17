import torch
import torch.nn as nn
from tokenizers import Tokenizer
from layers import Decoder, DecoderLayer, MultiheadAttention, FeedForward, Embedding


def get_pad_mask(x: torch.Tensor, pad_index: int):
    return (x != pad_index).unsqueeze(-2)


def get_subsequent_mask(x: torch.Tensor):
    batch_size, seq_len = x.size()
    mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool()

    return mask.unsqueeze(0).expand(batch_size, -1, -1)


class GeneratorTransformer(nn.Module):
    def __init__(self, vocab_size, tokenizer: Tokenizer, d_model=256, num_heads=8, d_ff=1024, num_layers=4, max_length=128, pad_index=0, dropout=0.1, device="cuda"):
        super().__init__()
        mha = MultiheadAttention(d_model, num_heads, dropout)
        ffn = FeedForward(d_model, d_ff, dropout)
        self.decoder = Decoder(DecoderLayer(mha, ffn, dropout), num_layers)
        self.embedding = Embedding(d_model, vocab_size, pad_index)
        self.norm = nn.LayerNorm(d_model)
        self.vocab_projection = nn.Linear(d_model, vocab_size)
        self.pad_index = pad_index
        self.context_len = max_length
        self.device = device
        self.tokenizer = tokenizer
        self.bos_token_id = tokenizer.token_to_id("<s>")
        self.eos_token_id = tokenizer.token_to_id("</s>")

    def forward(self, input_ids):
        mask = get_pad_mask(input_ids, self.pad_index) & get_subsequent_mask(input_ids)
        x = self.embedding(input_ids)
        x = self.decoder(x, mask)
        x = self.norm(x)
        logits = self.vocab_projection(x)

        return logits

    @torch.no_grad()
    def generate(self, prompt, temperature=1.0, max_out_tokens=200):
        self.eval()
        ids = self.tokenizer.encode(prompt).ids
        ids = [self.bos_token_id] + ids
        generated = torch.tensor([ids], dtype=torch.long, device=self.device)

        for _ in range(max_out_tokens):
            input_ids = generated[:, -self.context_len:]
            logits = self(input_ids)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            if next_token.item() == self.eos_token_id:
                break

        ids = generated[0].tolist()

        if ids and ids[0] == self.bos_token_id:
            ids = ids[1:]

        if self.eos_token_id in ids:
            ids = ids[:ids.index(self.eos_token_id)]

        return self.tokenizer.decode(ids)
    
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, tokenizer, vocab_size, pad_index, device="cpu"):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint["config"]

        model = cls(
            vocab_size=vocab_size,
            tokenizer=tokenizer,
            d_model=config["d_model"],
            num_heads=config["num_heads"],
            d_ff=config["d_ff"],
            num_layers=config["num_layers"],
            max_length=config["max_length"],
            pad_index=pad_index,
            dropout=config["dropout"],
            device=device
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device=device)
        model.eval()

        return model