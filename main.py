from cs336_basics.transformer import Transformer, AdamW, cross_entropy, gradient_clipping, torch_get_batch, save_checkpoint, decode as transformer_decode
from cs336_basics.bpe import Tokenizer, train_bpe
from pydantic import BaseModel
import argparse
import wandb
import torch
import os
from dotenv import load_dotenv
import json
import tokenizers as hf_tokenizers 
from tokenizers import \
    trainers as hf_trainers, models as hf_models, \
    pre_tokenizers as hf_pre_tokenizers
from tqdm import tqdm

def str_to_dtype(s: str) -> torch.dtype:
    if s == "float16":
        return torch.float16

    elif s == "bfloat16":
        return torch.bfloat16

    elif s == "float32":
        return torch.float32

    assert False, f"invalid dtype: {s}"

def print_validation(
    iteration: int,
    val_dataset: torch.Tensor,
    val_batch_size: int,
    context_length: int,
    model: torch.nn.Module,
    device: torch.device,
    offline: bool
):
    with torch.no_grad():
        val_inputs, val_targets = torch_get_batch(val_dataset, val_batch_size, context_length, device)
        val_outputs = model(val_inputs)
        val_loss = cross_entropy(val_outputs, val_targets)

        print(f"Iteration {iteration}, Val Loss: {val_loss.item()}")

        if not offline:
            wandb.log({
                "val_loss": val_loss.item(),
                "iteration": iteration
            })

def run_train_loop(
    project: str,
    run: str,
    description: str,
    checkpoint_enabled: bool,
    checkpoint_dir: str,
    checkpoint_prefix: str,
    train_dataset: torch.Tensor,
    val_dataset: torch.Tensor,
    hyperparams: dict,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_batch_size: int,
    val_batch_size: int,
    context_length: int,
    epochs: int,
    iterations_per_epoch: int,
    device: torch.device,
    offline: bool
):
    if checkpoint_enabled:
        assert os.path.isdir(checkpoint_dir), f"{checkpoint_dir} must be a directory"

    # Initialize wandb
    if not offline:
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        wandb.init(
            entity="rolph-recto-personal",
            project=project,
            id=run,
            name=description,
            config=hyperparams
        )

    print_validation(
        0,
        val_dataset=val_dataset,
        val_batch_size=val_batch_size,
        context_length=context_length,
        model=model,
        device=device,
        offline=offline
    )

    model.train()

    epoch: int = 0
    iteration: int = 0

    print("Beginning training loop")
    while epoch < epochs:
        print(f"Iteration {iteration+1}, Epoch {epoch+1}")

        optimizer.zero_grad()
        inputs, targets = torch_get_batch(train_dataset, train_batch_size, context_length, device)

        outputs = model(inputs)
        loss = cross_entropy(outputs, targets)
        loss.backward()
        gradient_clipping(model.parameters(), hyperparams["max_l2_norm"])
        optimizer.step()

        if not offline:
            wandb.log({
                "train_loss": loss.item(),
                "iteration": iteration+1
            })

        print(f"Iteration {iteration+1}, Train Loss: {loss.item()}")

        if iteration % iterations_per_epoch == 0:
            # compute validation loss
            print_validation(
                iteration,
                val_dataset=val_dataset,
                val_batch_size=val_batch_size,
                context_length=context_length,
                model=model,
                device=device,
                offline=offline
            )

            if checkpoint_enabled:
                checkpoint_filepath = os.path.join(checkpoint_dir, f"{checkpoint_prefix}{iteration}.pt")
                save_checkpoint(model, optimizer, iteration, checkpoint_filepath)
                print(f"Finished epoch {epoch+1}, saving checkpoint in {checkpoint_filepath}")

            epoch += 1

        iteration += 1

    print_validation(
        iteration,
        val_dataset=val_dataset,
        val_batch_size=val_batch_size,
        context_length=context_length,
        model=model,
        device=device,
        offline=offline
    )

    # Finish wandb run
    if not offline:
        wandb.finish()

def train_tokenizer(config: dict, args: argparse.Namespace):
    trainer = \
        hf_trainers.BpeTrainer(
            vocab_size=config["hyperparameters"]["vocab_size"],
            special_tokens=config["tokenizer"]["special_tokens"],
        )

    tokenizer = hf_tokenizers.Tokenizer(hf_models.BPE())
    tokenizer.train([args.dataset], trainer)
    tokenizer.save(config["tokenizer"]["file"])

    # special_tokens = config["tokenizer"]["special_tokens"]
    # vocab, merges = \
    #     train_bpe(args.dataset, config["hyperparameters"]["vocab_size"], special_tokens)

    # tokenizer = Tokenizer(vocab, merges, special_tokens)
    # tokenizer.to_files(config["tokenizer"]["vocab_file"], config["tokenizer"]["merges_file"])

def read_in_chunks(file_object, chunk_size=1024*1024):
  """Lazy function (generator) to read a file piece by piece.
  Default chunk size: 1k."""
  while True:
      data = file_object.read(chunk_size)
      if not data:
          break
      yield data

def tokenize_dataset(config: dict, args: argparse.Namespace):
    tokenizer_config = config["tokenizer"]
    tokenizer: hf_tokenizers.Tokenizer = hf_tokenizers.Tokenizer.from_file(tokenizer_config["file"])

    # tokenizer = Tokenizer.from_files2(
    #     vocab_filepath=tokenizer_config["vocab_file"],
    #     merges_filepath=tokenizer_config["merges_file"],
    #     special_tokens=tokenizer_config["special_tokens"]
    # )

    chunk_size = 1024 * 1024
    input_size = os.path.getsize(args.input_file)
    with open(args.input_file, "r") as f:
        tensors = []
        i = 1
        num_chunks = input_size // chunk_size + (0 if input_size % chunk_size == 0 else 1)
        for chunk in tqdm(read_in_chunks(f, chunk_size), total=num_chunks):
            encoded_input: hf_tokenizers.Encoding = tokenizer.encode(chunk)
            tensor_input = torch.tensor(encoded_input.ids)
            tensors.append(tensor_input)
            i += 1

        print(f"saving encoded input to {args.output_file}")
        torch.save(torch.cat(tensors), args.output_file)

def train(config: dict, args: argparse.Namespace):
    hyperparams = config["hyperparameters"]

    train_dataset = torch.load(config["train_dataset"])
    val_dataset = torch.load(config["val_dataset"])

    device = torch.device(config["device"])
    dtype = str_to_dtype(config["dtype"])

    model = Transformer(
        vocab_size=hyperparams["vocab_size"],
        context_length=hyperparams["context_length"],
        d_model=hyperparams["d_model"],
        num_layers=hyperparams["num_layers"],
        num_heads=hyperparams["num_heads"],
        d_ff=hyperparams["d_ff"],
        theta=hyperparams["theta"],
        device=device,
        dtype=dtype
    )

    optimizer = AdamW(
        params = model.parameters(),
        lr=hyperparams["lr"],
        weight_decay=hyperparams["weight_decay"],
        betas=(hyperparams["beta1"], hyperparams["beta2"]),
        eps=hyperparams["eps"]
   )

    run_train_loop(
        config["project"],
        config["run"],
        config["description"],
        config["checkpoint"]["enabled"],
        config["checkpoint"]["dir"],
        config["checkpoint"]["prefix"],
        train_dataset,
        val_dataset,
        hyperparams,
        model,
        optimizer,
        hyperparams["train_batch_size"],
        hyperparams["val_batch_size"],
        hyperparams["context_length"],
        config["epochs"],
        config["iterations_per_epoch"],
        device,
        config["offline"]
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
        device=torch.device(config["device"]),
        dtype=str_to_dtype(config["dtype"])
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
    load_dotenv()

    parser = argparse.ArgumentParser(description="CS336 Basics: Transformer training and inference")
    subparsers = parser.add_subparsers(dest="command", help="Command to run", required=True)

    # Train tookenizer command
    train_tokenizer_parser = subparsers.add_parser("train_tokenizer", help="Train a tokenizer")
    train_tokenizer_parser.add_argument("config", help="JSON config file path")
    train_tokenizer_parser.add_argument("dataset", help="Input dataset to train tokenizer")

    # Tokenize command
    tokenize_parser = subparsers.add_parser("tokenize", help="Tokenize a dataset")
    tokenize_parser.add_argument("config", help="JSON config file path")
    tokenize_parser.add_argument("input_file", help="Input text file to tokenize")
    tokenize_parser.add_argument("output_file", help="Output file for tokenized dataset")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("config", help="JSON config file path")

    # Decode command
    decode_parser = subparsers.add_parser("decode", help="Generate text from a model")
    decode_parser.add_argument("config", help="JSON config file path")
    decode_parser.add_argument("checkpoint", help="Model checkpoint file path")
    decode_parser.add_argument("--prompt", "-p", required=True, help="Prompt text for generation")
    decode_parser.add_argument("--temperature", "-t", type=float, default=1.0, help="Sampling temperature")
    decode_parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling parameter")
    decode_parser.add_argument("--max-tokens", type=int, default=-1, help="Maximum tokens to generate")

    args = parser.parse_args()

    # Load config file
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Execute command
    if args.command == "train_tokenizer":
        train_tokenizer(config, args)
    elif args.command == "tokenize":
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
