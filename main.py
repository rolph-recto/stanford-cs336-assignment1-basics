from cs336_basics.transformer import *
from cs336_basics.bpe import Tokenizer
from pydantic import BaseModel
import argparse
import wandb

def run_train_loop(
    checkpoint_dir: str,
    checkpoint_prefix: str,
    dataset: torch.Tensor,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    context_length: int,
    max_l2_norm: float,
    epochs: int,
    iterations_per_epoch: int,
    device: str
):
    assert os.path.isdir(checkpoint_dir), f"{checkpoint_dir} must be a directory"

    model.train()
    iteration: int = 0
    for epoch in range(1, epochs+1):
        print(f"Epoch {epoch}")

        for iteration in range(1, iterations_per_epoch+1):
            print("iteration {i} of {iterations_per_epoch} of epoch {epoch}")

            optimizer.zero_grad()
            inputs, targets = torch_get_batch(dataset, batch_size, context_length, device)
            outputs = model(inputs)
            loss = cross_entropy(outputs, targets)
            loss.backward()
            gradient_clipping(model.parameters(), max_l2_norm)
            optimizer.step()

            iteration += 1

        checkpoint_filepath = os.path.join(checkpoint_dir, f"{checkpoint_prefix}{iteration}.pt")
        save_checkpoint(model, optimizer, iteration, checkpoint_filepath)
        print(f"Finished epoch {epoch}, saving checkpoint in {checkpoint_filepath}")

def tokenize_dataset(config: dict, args: argparse.Namespace):
    tokenizer_config = config["tokenizer"]
    tokenizer = Tokenizer.from_files(
        vocab_filepath=tokenizer_config["vocab_file"],
        merges_filepath=tokenizer_config["merges_file"],
        special_tokens=tokenizer_config["special_tokens"]
    )

    with open(args.input_file, "r") as f:
        dataset = torch.tensor(tokenizer.encode(f.read()))
        torch.save(dataset, args.output_file)

def train(config: dict, args: argparse.Namespace):
    hyperparams = config["hyperparameters"]

    dataset = torch.from_file(config["dataset"])

    model = Transformer(
        vocab_size=hyperparams["vocab_size"],
        context_length=hyperparams["context_length"],
        d_model=hyperparams["d_model"],
        num_layers=hyperparams["num_layers"],
        num_heads=hyperparams["num_heads"],
        d_ff=hyperparams["d_ff"],
        theta=hyperparams["theta"],
        device=config["device"]
    )

    optimizer = AdamW(
        params = model.parameters(),
        lr=hyperparams["lr"],
        weight_decay=hyperparams["weight_decay"],
        betas=(hyperparams["beta1"], hyperparams["beta2"]),
        eps=hyperparams["eps"]
   )

    run_train_loop(
        config["checkpoint"]["dir"],
        config["checkpoint"]["prefix"],
        dataset,
        model,
        optimizer,
        hyperparams["batch_size"],
        hyperparams["context_length"],
        hyperparams["max_l2_norm"],
        config["epochs"],
        config["iterations_per_epoch"],
        config["device"]
    )

def main():
    pass
