import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from tqdm import tqdm


class TransformerTrainer:
    """
    trainer class for Transformer models.
    handles training, evaluation, and visualization.
    """

    def __init__(self, model, train_loader, test_loader, config, device=None):
        """
        initialize the trainer.

        Args:
            model: the transformer model
            train_loader: DataLoader for training data
            test_loader: DataLoader for test data
            config: dictionary containing training hyperparameters
            device: device to use for training (cpu or cuda)
        """
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config

        # set device
        if device is None:
            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if hasattr(torch, "mps") and torch.mps.is_available()
                else "cpu"
            )
        else:
            self.device = device

        print(f"Using device: {self.device}")

        # move model to device
        self.model = self.model.to(self.device)

        # set up loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config["learning_rate"])

        # learning rate scheduler (linear warm-up followed by decay)
        self.scheduler = self._get_scheduler()

        # initialize tracking variables
        self.train_losses = []
        self.test_losses = []
        self.train_accs = []
        self.test_accs = []
        self.best_test_acc = 0.0

    def _get_scheduler(self):
        """create a learning rate scheduler with warmup"""
        warmup_steps = self.config.get("warmup_steps", 0)
        total_steps = self.config["epochs"] * len(self.train_loader)

        def lr_lambda(current_step):
            # linear warmup followed by inverse square root decay
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return (warmup_steps / current_step) ** 0.5

        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train_epoch(self, epoch):
        """train the model for one epoch"""
        self.model.train()
        epoch_loss = 0
        correct = 0
        total = 0

        # progress bar for training
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")

        for batch in progress_bar:
            # get inputs and labels
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["label"].to(self.device)

            # zero gradients
            self.optimizer.zero_grad()

            # forward pass
            logits, _ = self.model(input_ids)

            # calculate loss
            loss = self.criterion(logits, labels)

            # backward pass and optimize
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # update progress
            epoch_loss += loss.item()

            # calculate accuracy
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # update progress bar
            progress_bar.set_postfix(
                {
                    "loss": f"{epoch_loss / (progress_bar.n + 1):.4f}",
                    "acc": f"{100 * correct / total:.2f}%",
                }
            )

        # calculate average loss and accuracy for the epoch
        epoch_loss /= len(self.train_loader)
        epoch_acc = 100 * correct / total

        self.train_losses.append(epoch_loss)
        self.train_accs.append(epoch_acc)

        return epoch_loss, epoch_acc

    def evaluate(self):
        """evaluate the model on the test set"""
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in self.test_loader:
                # get inputs and labels
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["label"].to(self.device)

                # forward pass
                logits, _ = self.model(input_ids)

                # calculate loss
                loss = self.criterion(logits, labels)
                test_loss += loss.item()

                # calculate accuracy
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # store predictions and labels for metrics
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # calculate average loss and accuracy
        test_loss /= len(self.test_loader)
        test_acc = 100 * correct / total

        self.test_losses.append(test_loss)
        self.test_accs.append(test_acc)

        # update best accuracy
        if test_acc > self.best_test_acc:
            self.best_test_acc = test_acc

        return test_loss, test_acc, np.array(all_preds), np.array(all_labels)

    def train(self, output_dir=None):
        """
        train the model for the specified number of epochs.

        Args:
            output_dir: directory to save outputs (plots, models)

        Returns:
            training_stats: dictionary with training statistics
        """
        print(f"Starting training for {self.config['epochs']} epochs...")
        start_time = time.time()

        # create output directory if specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            model_dir = os.path.join(output_dir, "models")
            os.makedirs(model_dir, exist_ok=True)

        for epoch in range(self.config["epochs"]):
            # train one epoch
            train_loss, train_acc = self.train_epoch(epoch)

            # evaluate on test set
            test_loss, test_acc, _, _ = self.evaluate()

            print(
                f"Epoch {epoch + 1}/{self.config['epochs']} - "
                f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.2f}% - "
                f"Test loss: {test_loss:.4f}, Test acc: {test_acc:.2f}%"
            )

            # save model checkpoint
            if output_dir and test_acc == self.best_test_acc:
                model_path = os.path.join(model_dir, f"model_epoch_{epoch + 1}.pt")
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "train_loss": train_loss,
                        "test_loss": test_loss,
                        "train_acc": train_acc,
                        "test_acc": test_acc,
                    },
                    model_path,
                )
                print(f"Saved best model checkpoint to {model_path}")

        # calculate training time
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")

        # plot training curves
        if output_dir:
            self.plot_training_curves(output_dir)

        # final evaluation
        test_loss, test_acc, all_preds, all_labels = self.evaluate()
        print(f"Final test accuracy: {test_acc:.2f}%")

        # confusion matrix
        if output_dir:
            self.plot_confusion_matrix(all_preds, all_labels, output_dir)

        # return training statistics
        training_stats = {
            "train_losses": self.train_losses,
            "test_losses": self.test_losses,
            "train_accs": self.train_accs,
            "test_accs": self.test_accs,
            "best_test_acc": self.best_test_acc,
            "training_time": training_time,
        }

        return training_stats

    def plot_training_curves(self, output_dir):
        """Plot training and validation loss/accuracy curves"""
        epochs = range(1, len(self.train_losses) + 1)

        plt.figure(figsize=(12, 5))

        # plot loss curves
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_losses, "b-", label="Training Loss")
        plt.plot(epochs, self.test_losses, "r-", label="Validation Loss")
        plt.title("Training and Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()

        # plot accuracy curves
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.train_accs, "b-", label="Training Accuracy")
        plt.plot(epochs, self.test_accs, "r-", label="Validation Accuracy")
        plt.title("Training and Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy (%)")
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "training_curves.png"))
        plt.close()

    def plot_confusion_matrix(self, preds, labels, output_dir):
        """plot confusion matrix for binary classification"""
        from sklearn.metrics import confusion_matrix
        import seaborn as sns

        # create confusion matrix
        cm = confusion_matrix(labels, preds)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
        plt.close()

    def visualize_attention(self, text, output_dir=None):
        """
        visualize attention weights for a given input text.

        Args:
            text: input text to visualize attention for
            output_dir: directory to save the visualization

        Returns:
            attention_weights: list of attention weight matrices
        """
        from utils.data_utils import TextDataset

        # tokenize and encode text
        dataset = TextDataset(
            [text],
            vocab=self.train_loader.dataset.vocab,
            max_seq_len=self.config["max_seq_len"],
            build_vocab=False,
        )

        # get input_ids
        input_ids = dataset[0]["input_ids"].unsqueeze(0).to(self.device)

        # set model to evaluation mode
        self.model.eval()

        # forward pass
        with torch.no_grad():
            _, attention_weights = self.model(input_ids)

        # get tokens for visualization
        tokens = dataset.vocab.convert_ids_to_tokens(input_ids[0].cpu().numpy())

        # only keep non-padding tokens
        pad_idx = dataset.vocab.token2idx["<PAD>"]
        non_pad_mask = input_ids[0] != pad_idx
        tokens = [tokens[i] for i in range(len(tokens)) if non_pad_mask[i]]

        # visualize attention weights from the last layer's first head
        last_layer_attn = (
            attention_weights[-1][0, 0, : len(tokens), : len(tokens)].cpu().numpy()
        )

        plt.figure(figsize=(10, 8))
        plt.imshow(last_layer_attn, cmap="viridis")
        plt.colorbar()
        plt.xticks(range(len(tokens)), tokens, rotation=90)
        plt.yticks(range(len(tokens)), tokens)
        plt.xlabel("Key")
        plt.ylabel("Query")
        plt.title(f'Attention Weights for: "{text}"')

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(
                os.path.join(output_dir, "attention_visualization.png"),
                bbox_inches="tight",
            )
            plt.close()
        else:
            plt.show()

        return attention_weights
