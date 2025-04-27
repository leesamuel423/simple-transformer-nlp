import torch
import argparse
import os
import json
from models.transformer import TransformerForSequenceClassification
from utils.data_utils import create_data_loaders, load_sample_data
from utils.training_utils import TransformerTrainer
from configs.model_config import CONFIG
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_recall_fscore_support,
)


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

    return model, checkpoint


def evaluate(model, test_loader, device):
    """evaluate model on test data"""
    model.eval()
    model = model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)

            logits, _ = model(input_ids)
            _, predicted = torch.max(logits, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted"
    )

    # detailed classification report
    report = classification_report(all_labels, all_preds, digits=4)

    return accuracy, precision, recall, f1, report, all_preds, all_labels


def main(args):
    # set device
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if hasattr(torch, "mps") and torch.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # create output directory
    eval_dir = os.path.join(args.model_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

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

    # find best model checkpoint
    model_files = [
        f
        for f in os.listdir(os.path.join(args.model_dir, "models"))
        if f.endswith(".pt")
    ]

    if not model_files:
        print("No model checkpoints found!")
        return

    best_model_path = os.path.join(args.model_dir, "models", model_files[-1])
    print(f"Loading model from: {best_model_path}")

    # load model
    model, checkpoint = load_model(best_model_path, CONFIG, args.num_classes)
    print(
        f"Loaded model from epoch {checkpoint['epoch']} with "
        f"test accuracy: {checkpoint['test_acc']:.2f}%"
    )

    # evaluate model
    print("Evaluating model...")
    accuracy, precision, recall, f1, report, all_preds, all_labels = evaluate(
        model, test_loader, device
    )

    print("\nEvaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(report)

    # save results
    results = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "classification_report": report,
    }

    with open(os.path.join(eval_dir, "evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=4)

    # create trainer for visualization
    trainer = TransformerTrainer(
        model=model, train_loader=train_loader, test_loader=test_loader, config=CONFIG
    )

    # visualize attention for all test examples
    print("\nVisualizing attention for test examples...")
    for i, text in enumerate(test_texts):
        trainer.visualize_attention(
            text, output_dir=os.path.join(eval_dir, f"attention_viz_example_{i}")
        )

    print(f"\nEvaluation completed! Results saved to {eval_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained Transformer model")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="results",
        help="Directory containing the trained model",
    )
    parser.add_argument(
        "--num_classes", type=int, default=2, help="Number of output classes"
    )
    args = parser.parse_args()

    main(args)
