import torch
import argparse
import os
from models.transformer import TransformerForSequenceClassification
from utils.data_utils import create_data_loaders, load_sample_data
from utils.training_utils import TransformerTrainer
from configs.model_config import CONFIG


def main(args):
    # set random seed for reproducibility
    torch.manual_seed(args.seed)

    # create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # load data
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

    print(f"Vocabulary size: {CONFIG['vocab_size']}")
    print(f"Training samples: {len(train_texts)}")
    print(f"Test samples: {len(test_texts)}")

    # create model
    print("Creating model...")
    model = TransformerForSequenceClassification(
        vocab_size=CONFIG["vocab_size"],
        d_model=CONFIG["d_model"],
        num_heads=CONFIG["n_heads"],
        num_layers=CONFIG["n_layers"],
        d_ff=CONFIG["d_ff"],
        max_seq_len=CONFIG["max_seq_len"],
        num_classes=args.num_classes,
        dropout=CONFIG["dropout"],
    )

    # create trainer
    trainer = TransformerTrainer(
        model=model, train_loader=train_loader, test_loader=test_loader, config=CONFIG
    )

    # train model
    print("Starting training...")
    trainer.train(output_dir=args.output_dir)

    # visualize attention for a sample text
    print("\nVisualizing attention for a sample text...")
    sample_text = "This movie was fantastic! I really enjoyed it."
    trainer.visualize_attention(sample_text, output_dir=args.output_dir)

    print(f"\nTraining completed! All outputs saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Transformer model for text classification"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results", help="Directory to save results"
    )
    parser.add_argument(
        "--num_classes", type=int, default=2, help="Number of output classes"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    main(args)
