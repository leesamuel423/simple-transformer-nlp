import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.transformer import TransformerEncoder, TransformerForSequenceClassification
from configs.model_config import CONFIG

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_outputs")
os.makedirs(output_dir, exist_ok=True)


def test_transformer_encoder():
    print("Testing Transformer Encoder...")

    # parameters
    batch_size = 2
    seq_len = 10
    vocab_size = CONFIG["vocab_size"]
    d_model = CONFIG["d_model"]
    num_heads = CONFIG["n_heads"]
    num_layers = CONFIG["n_layers"]
    d_ff = CONFIG["d_ff"]
    max_seq_len = CONFIG["max_seq_len"]

    # create random input indices
    src = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"Input shape: {src.shape}")

    # create a sample mask
    src_mask = torch.ones(batch_size, 1, 1, seq_len)
    src_mask[:, :, :, 0] = 0  # First token is masked

    # initialize transformer encoder
    transformer = TransformerEncoder(
        vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len
    )

    # forward pass
    output, attention_weights = transformer(src, src_mask)
    print(f"Output shape: {output.shape}")
    print(f"Number of attention weight tensors: {len(attention_weights)}")

    # expected output shape
    expected_shape = (batch_size, seq_len, d_model)
    assert output.shape == expected_shape, (
        f"Expected output shape {expected_shape}, got {output.shape}"
    )

    # visualize attention weights from the first head of the last layer
    plt.figure(figsize=(10, 8))
    plt.imshow(attention_weights[-1][0, 0].detach().numpy(), cmap="viridis")
    plt.colorbar()
    plt.xlabel("Key positions")
    plt.ylabel("Query positions")
    plt.title("Transformer Encoder Final Layer Attention")

    # add position labels
    plt.xticks(np.arange(seq_len))
    plt.yticks(np.arange(seq_len))

    output_path = os.path.join(output_dir, "transformer_encoder_attention.png")
    plt.savefig(output_path)
    print(f"Transformer encoder attention visualization saved to {output_path}")

    print("Transformer Encoder test completed!\n")

    return output, attention_weights


def test_transformer_for_classification():
    print("Testing Transformer for Sequence Classification...")

    # parameters
    batch_size = 2
    seq_len = 10
    vocab_size = CONFIG["vocab_size"]
    d_model = CONFIG["d_model"]
    num_heads = CONFIG["n_heads"]
    num_layers = CONFIG["n_layers"]
    d_ff = CONFIG["d_ff"]
    max_seq_len = CONFIG["max_seq_len"]
    num_classes = 3  # Example: sentiment analysis (negative, neutral, positive)

    # create random input indices
    src = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"Input shape: {src.shape}")

    # initialize transformer for sequence classification
    model = TransformerForSequenceClassification(
        vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len, num_classes
    )

    # set model to evaluation mode
    model.eval()

    # forward pass
    with torch.no_grad():
        logits, attention_weights = model(src)

    print(f"Logits shape: {logits.shape}")
    print(f"Number of attention weight tensors: {len(attention_weights)}")

    # expected logits shape
    expected_shape = (batch_size, num_classes)
    assert logits.shape == expected_shape, (
        f"Expected logits shape {expected_shape}, got {logits.shape}"
    )

    # apply softmax to get probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1)
    print("Class probabilities for batch samples:")
    for i in range(batch_size):
        print(f"  Sample {i}: {probs[i].tolist()}")

    # visualize attention weights from first head of the last layer
    plt.figure(figsize=(10, 8))
    plt.imshow(attention_weights[-1][0, 0].detach().numpy(), cmap="viridis")
    plt.colorbar()
    plt.xlabel("Key positions")
    plt.ylabel("Query positions")
    plt.title("Transformer Classification Model Attention")

    # add position labels
    plt.xticks(np.arange(seq_len))
    plt.yticks(np.arange(seq_len))

    output_path = os.path.join(output_dir, "transformer_classification_attention.png")
    plt.savefig(output_path)
    print(f"Transformer classification attention visualization saved to {output_path}")

    print("Transformer for Sequence Classification test completed!")

    return logits, attention_weights


if __name__ == "__main__":
    # test the transformer components
    test_transformer_encoder()
    test_transformer_for_classification()

    print("\nAll transformer tests completed successfully!")
