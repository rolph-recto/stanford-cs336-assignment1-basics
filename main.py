from cs336_basics.transformer import Transformer, AdamW, cross_entropy, gradient_clipping, torch_get_batch, save_checkpoint, decode as transformer_decode
from cs336_basics.bpe import Tokenizer
from pydantic import BaseModel
import argparse
import wandb
import torch
import os
import json

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

    dataset = torch.load(config["dataset"])

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

def decode(config: dict, args: argparse.Namespace):
    # Load tokenizer
    tokenizer_config = config["tokenizer"]
    tokenizer = Tokenizer.from_files(
        vocab_filepath=tokenizer_config["vocab_file"],
        merges_filepath=tokenizer_config["merges_file"],
        special_tokens=tokenizer_config["special_tokens"]
    )

    # Load model from checkpoint
    checkpoint = torch.load(args.checkpoint)
    hyperparams = config["hyperparameters"]

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

    model.load_state_dict(checkpoint["model"])
    model.eval()

    # Get decoding parameters from args or config
    temperature = args.temperature if hasattr(args, 'temperature') else config.get("decode", {}).get("temperature", 1.0)
    top_p = args.top_p if hasattr(args, 'top_p') else config.get("decode", {}).get("top_p", 1.0)
    max_tokens = args.max_tokens if hasattr(args, 'max_tokens') else config.get("decode", {}).get("max_tokens", -1)

    # Decode
    output = transformer_decode(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        context_length=hyperparams["context_length"],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p
    )

    print(output)

def main():
    parser = argparse.ArgumentParser(description="CS336 Basics: Transformer training and inference")
    subparsers = parser.add_subparsers(dest="command", help="Command to run", required=True)

    # Tokenize command
    tokenize_parser = subparsers.add_parser("tokenize", help="Tokenize a dataset")
    tokenize_parser.add_argument("config", required=True, help="JSON config file path")
    tokenize_parser.add_argument("input_file", required=True, help="Input text file to tokenize")
    tokenize_parser.add_argument("output_file", required=True, help="Output file for tokenized dataset")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("config", required=True, help="JSON config file path")

    # Decode command
    decode_parser = subparsers.add_parser("decode", help="Generate text from a model")
    decode_parser.add_argument("config", required=True, help="JSON config file path")
    decode_parser.add_argument("checkpoint", required=True, help="Model checkpoint file path")
    decode_parser.add_argument("--prompt", "-p", required=True, help="Prompt text for generation")
    decode_parser.add_argument("--temperature", "-t", type=float, default=1.0, help="Sampling temperature")
    decode_parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling parameter")
    decode_parser.add_argument("--max-tokens", type=int, default=-1, help="Maximum tokens to generate")

    args = parser.parse_args()

    # Load config file
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Execute command
    if args.command == "tokenize":
        tokenize_dataset(config, args)
    elif args.command == "train":
        train(config, args)
    elif args.command == "decode":
        decode(config, args)
    else:
        parser.print_help()
        exit(1)

if __name__ == "__main__":
    main()
