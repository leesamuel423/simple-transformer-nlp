import torch
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from models.transformer import TransformerForSequenceClassification
from utils.data_utils import TextDataset, load_sample_data, create_data_loaders
from configs.model_config import CONFIG


def load_model(model_path, config, num_classes):
    """load a trained model from checkpoint"""
    model = TransformerForSequenceClassification(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        num_heads=config["n_heads"],
        num_layers=config["n_layers"],
        d_ff=config["d_ff"],
        max_seq_len=config["max_seq_len"],
        num_classes=num_classes,
        dropout=config["dropout"],
    )

    # load model state
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


def visualize_attention_heads(model, text, vocab, output_dir):
    """Visualize attention weights for all heads in the last layer"""
    # tokenize and encode text
    dataset = TextDataset(
        [text], vocab=vocab, max_seq_len=CONFIG["max_seq_len"], build_vocab=False
    )
    input_ids = dataset[0]["input_ids"].unsqueeze(0)

    # get attention weights
    with torch.no_grad():
        _, attention_weights = model(input_ids)

    # get tokens
    tokens = vocab.convert_ids_to_tokens(input_ids[0].numpy())

    # remove padding tokens
    pad_idx = vocab.token2idx["<PAD>"]
    non_pad_mask = input_ids[0] != pad_idx
    tokens = [tokens[i] for i in range(len(tokens)) if non_pad_mask[i]]

    # get last layer attention weights
    last_layer_attn = attention_weights[-1]  # [batch_size, num_heads, seq_len, seq_len]

    # number of heads
    num_heads = last_layer_attn.shape[1]

    # create subplot grid
    n_cols = 2
    n_rows = (num_heads + n_cols - 1) // n_cols

    plt.figure(figsize=(n_cols * 6, n_rows * 5))
    plt.suptitle(f'Attention Heads for: "{text}"', fontsize=16)

    for h in range(num_heads):
        # get attention weights for this head
        head_attn = last_layer_attn[0, h, : len(tokens), : len(tokens)].numpy()

        # create subplot
        plt.subplot(n_rows, n_cols, h + 1)
        plt.imshow(head_attn, cmap="viridis")
        plt.colorbar()
        plt.xticks(range(len(tokens)), tokens, rotation=90)
        plt.yticks(range(len(tokens)), tokens)
        plt.xlabel("Key")
        plt.ylabel("Query")
        plt.title(f"Head {h + 1}")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(output_dir, "attention_heads.png"), bbox_inches="tight")
    plt.close()


def visualize_attention_layer_comparison(model, text, vocab, output_dir):
    """compare attention patterns across different layers"""
    # tokenize and encode text
    dataset = TextDataset(
        [text], vocab=vocab, max_seq_len=CONFIG["max_seq_len"], build_vocab=False
    )
    input_ids = dataset[0]["input_ids"].unsqueeze(0)

    # get attention weights
    with torch.no_grad():
        _, attention_weights = model(input_ids)

    # get tokens
    tokens = vocab.convert_ids_to_tokens(input_ids[0].numpy())

    # remove padding tokens
    pad_idx = vocab.token2idx["<PAD>"]
    non_pad_mask = input_ids[0] != pad_idx
    tokens = [tokens[i] for i in range(len(tokens)) if non_pad_mask[i]]

    # num of layers
    num_layers = len(attention_weights)

    # create subplot grid
    plt.figure(figsize=(15, 5 * num_layers))
    plt.suptitle(f'Attention Across Layers for: "{text}"', fontsize=16)

    for layer in range(num_layers):
        # get average attention across heads for this layer
        layer_attn = attention_weights[layer][0]  # [num_heads, seq_len, seq_len]
        avg_attn = layer_attn.mean(dim=0)[: len(tokens), : len(tokens)].numpy()

        # create subplot
        plt.subplot(num_layers, 1, layer + 1)
        plt.imshow(avg_attn, cmap="viridis")
        plt.colorbar()
        plt.xticks(range(len(tokens)), tokens, rotation=90)
        plt.yticks(range(len(tokens)), tokens)
        plt.xlabel("Key")
        plt.ylabel("Query")
        plt.title(f"Layer {layer + 1}")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(output_dir, "attention_layers.png"), bbox_inches="tight")
    plt.close()


def visualize_token_importance(model, text, vocab, output_dir):
    """visualize the importance of each token based on attention"""
    # tokenize and encode text
    dataset = TextDataset(
        [text], vocab=vocab, max_seq_len=CONFIG["max_seq_len"], build_vocab=False
    )
    input_ids = dataset[0]["input_ids"].unsqueeze(0)

    # get attention weights
    with torch.no_grad():
        _, attention_weights = model(input_ids)

    # get tokens
    tokens = vocab.convert_ids_to_tokens(input_ids[0].numpy())

    # Remove padding tokens
    pad_idx = vocab.token2idx["<PAD>"]
    non_pad_mask = input_ids[0] != pad_idx
    tokens = [tokens[i] for i in range(len(tokens)) if non_pad_mask[i]]

    # get last layer attention weights
    last_layer_attn = attention_weights[-1][0]  # shape: [num_heads, seq_len, seq_len]

    # average across heads
    avg_attn = last_layer_attn.mean(dim=0)[: len(tokens), : len(tokens)]

    # calculate token importance (sum of attention received by each token)
    token_importance = avg_attn.sum(dim=0).numpy()

    # normalize
    token_importance = token_importance / token_importance.max()

    # create horizontal bar chart
    plt.figure(figsize=(10, max(5, len(tokens) * 0.5)))
    plt.barh(range(len(tokens)), token_importance, color="skyblue")
    plt.yticks(range(len(tokens)), tokens)
    plt.xlabel("Relative Importance")
    plt.ylabel("Token")
    plt.title(f'Token Importance for: "{text}"')
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "token_importance.png"))
    plt.close()


def visualize_positional_attention(model, text, vocab, output_dir):
    """visualize how attention changes with token position"""
    # tokenize and encode text
    dataset = TextDataset(
        [text], vocab=vocab, max_seq_len=CONFIG["max_seq_len"], build_vocab=False
    )
    input_ids = dataset[0]["input_ids"].unsqueeze(0)

    # get attention weights
    with torch.no_grad():
        _, attention_weights = model(input_ids)

    # get tokens
    tokens = vocab.convert_ids_to_tokens(input_ids[0].numpy())

    # remove padding tokens
    pad_idx = vocab.token2idx["<PAD>"]
    non_pad_mask = input_ids[0] != pad_idx
    tokens = [tokens[i] for i in range(len(tokens)) if non_pad_mask[i]]
    n_tokens = len(tokens)

    # get last layer attention weights (averaged across heads)
    last_layer_attn = attention_weights[-1][0].mean(dim=0)[:n_tokens, :n_tokens].numpy()

    # create heatmap with token positions
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(
        last_layer_attn,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        xticklabels=tokens,
        yticklabels=tokens,
    )
    plt.xlabel("Attended Token")
    plt.ylabel("Attending Token")
    plt.title(f'Attention Matrix for: "{text}"')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "positional_attention.png"))
    plt.close()


def main(args):
    # create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # load data to get vocabulary
    print("Loading data...")
    train_texts, train_labels, test_texts, test_labels = load_sample_data()

    # create data loaders
    train_loader, test_loader, vocab = create_data_loaders(
        train_texts,
        train_labels,
        test_texts,
        test_labels,
        batch_size=CONFIG["batch_size"],
        max_seq_len=CONFIG["max_seq_len"],
    )

    # update vocabulary size in config
    CONFIG["vocab_size"] = len(vocab)

    # load model
    if args.model_path:
        model_path = args.model_path
    else:
        # find latest model checkpoint
        model_dir = os.path.join(args.results_dir, "models")
        model_files = [f for f in os.listdir(model_dir) if f.endswith(".pt")]

        if not model_files:
            print("No model checkpoints found!")
            return

        model_path = os.path.join(model_dir, model_files[-1])

    print(f"Loading model from: {model_path}")
    model, _ = load_model(model_path, CONFIG, args.num_classes)

    # process the input text
    input_text = (
        args.text if args.text else "This movie was fantastic! I really enjoyed it."
    )

    print(f'Visualizing attention for: "{input_text}"')

    # create visualizations
    viz_dir = os.path.join(args.output_dir, "attention_visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    print("Generating visualizations...")

    # generate different types of visualizations
    visualize_attention_heads(model, input_text, vocab, viz_dir)
    visualize_attention_layer_comparison(model, input_text, vocab, viz_dir)
    visualize_token_importance(model, input_text, vocab, viz_dir)
    visualize_positional_attention(model, input_text, vocab, viz_dir)

    print(f"Visualizations saved to {viz_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize attention in Transformer model"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory containing the training results",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to a specific model checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="visualizations",
        help="Directory to save visualizations",
    )
    parser.add_argument(
        "--text", type=str, default=None, help="Text to visualize attention for"
    )
    parser.add_argument(
        "--num_classes", type=int, default=2, help="Number of output classes"
    )
    args = parser.parse_args()

    main(args)
