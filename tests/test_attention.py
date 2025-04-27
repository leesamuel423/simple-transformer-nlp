import torch
import matplotlib.pyplot as plt
import sys
import os

# Add the project root to the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.attention import ScaledDotProductAttention, MultiHeadAttention
from configs.model_config import CONFIG

# Create output directory if it doesn't exist
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_outputs")
os.makedirs(output_dir, exist_ok=True)


def test_scaled_dot_product_attention():
    print("Testing Scaled Dot-Product Attention...")

    # sample inputs
    batch_size = 2
    num_heads = 1
    seq_len = 4
    d_k = 8

    # random query, key, value tensors
    query = torch.randn(batch_size, num_heads, seq_len, d_k)
    key = torch.randn(batch_size, num_heads, seq_len, d_k)
    value = torch.randn(batch_size, num_heads, seq_len, d_k)

    # sample mask (first token attends to nothing)
    mask = torch.ones(batch_size, 1, 1, seq_len)
    mask[:, :, :, 0] = 0

    # initialize attention layer
    attention = ScaledDotProductAttention()

    # forward pass
    output, attention_weights = attention(query, key, value, mask)

    print(f"Query shape: {query.shape}")
    print(f"Key shape: {key.shape}")
    print(f"Value shape: {value.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")

    # check that masked position has zero attention
    print(
        f"First token attention weights (should be 0): {attention_weights[0, 0, 0, 0]}"
    )

    # visualize attention weights
    plt.figure(figsize=(8, 6))
    plt.imshow(attention_weights[0, 0].detach().numpy(), cmap="viridis")
    plt.colorbar()
    plt.xlabel("Key positions")
    plt.ylabel("Query positions")
    plt.title("Attention Weights")
    output_path = os.path.join(output_dir, "attention_weights.png")
    plt.savefig(output_path)
    print(f"Attention weights visualization saved to {output_path}")
    print("Scaled Dot-Product Attention test completed!\n")
    return output, attention_weights


def test_multi_head_attention():
    print("Testing Multi-Head Attention...")

    # parameters
    batch_size = 2
    seq_len = 4
    d_model = CONFIG["d_model"]
    num_heads = CONFIG["n_heads"]

    # random input tensors
    query = torch.randn(batch_size, seq_len, d_model)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)

    # sample mask
    mask = torch.ones(batch_size, 1, 1, seq_len)
    mask[:, :, :, 0] = 0  # First token is masked

    # initialize multi-head attention
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

    # forward pass
    output, attention_weights = mha(query, key, value, mask)

    print(f"Query shape: {query.shape}")
    print(f"Key shape: {key.shape}")
    print(f"Value shape: {value.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")

    # check that output has the expected shape
    expected_shape = (batch_size, seq_len, d_model)
    assert output.shape == expected_shape, (
        f"Expected output shape {expected_shape}, got {output.shape}"
    )

    # visualize one head's attention weights
    plt.figure(figsize=(8, 6))
    plt.imshow(attention_weights[0, 0].detach().numpy(), cmap="viridis")
    plt.colorbar()
    plt.xlabel("Key positions")
    plt.ylabel("Query positions")
    plt.title("Multi-Head Attention Weights (Head 0)")
    output_path = os.path.join(output_dir, "multihead_attention_weights.png")
    plt.savefig(output_path)
    print(f"Multi-Head Attention weights visualization saved to {output_path}")
    print("Multi-Head Attention test completed!")
    return output, attention_weights


if __name__ == "__main__":
    # test the attention mechanisms
    test_scaled_dot_product_attention()
    test_multi_head_attention()

    print("\nAll tests completed successfully!")
