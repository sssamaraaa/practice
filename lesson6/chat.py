import torch
from tokenizers import Tokenizer
from dataset import TextDataset
from transformer import GeneratorTransformer


def chat():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.from_file("lesson6\mistral_tokenizer.json")
    tokenizer.add_special_tokens(["<pad>", "<s>", "</s>"])

    model = GeneratorTransformer.load_from_checkpoint(
        "lesson6/checkpoints/epoch_23.pt",
        tokenizer=tokenizer,
        vocab_size=tokenizer.get_vocab_size(),
        pad_index=tokenizer.token_to_id("<pad>"),
        device=device
    )

    while True:
        user_input = input("Вы: ")
        if user_input.lower() == "quit":
            break

        response = model.generate(user_input, temperature=0.8, max_out_tokens=100)
        print("Бот:", response)


if __name__ == "__main__":
    chat()