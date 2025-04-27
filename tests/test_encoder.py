import torch
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.encoder import LayerNorm, FeedForward, EncoderLayer, Encoder
from configs.model_config import CONFIG

# create output directory if it doesn't exist
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_outputs")
os.makedirs(output_dir, exist_ok=True)


def test_layer_norm():
    print("Testing Layer Normalization...")

    # sample input
    batch_size = 2
    seq_len = 4
    d_model = CONFIG["d_model"]

    # random tensor
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"Input shape: {x.shape}")

    # compute mean and std of the input
    x_mean = x.mean(dim=-1, keepdim=True)
    x_std = x.std(dim=-1, keepdim=True)
    print(f"Input mean: {x_mean.mean().item():.6f}")
    print(f"Input std: {x_std.mean().item():.6f}")

    # initialize layer norm
    layer_norm = LayerNorm(d_model)

    # forward pass
    output = layer_norm(x)
    print(f"Output shape: {output.shape}")

    # compute mean and std of the output
    output_mean = output.mean(dim=-1, keepdim=True)
    output_std = output.std(dim=-1, keepdim=True)
    print(f"Output mean: {output_mean.mean().item():.6f}")
    print(f"Output std: {output_std.mean().item():.6f}")

    # check if the output has been normalized
    assert abs(output_mean.mean().item()) < 1e-5, "Output mean should be close to 0"
    assert abs(output_std.mean().item() - 1.0) < 1e-1, "Output std should be close to 1"

    print("Layer Normalization test completed!\n")

    return output


def test_feed_forward():
    print("Testing Feed-Forward Network...")

    # parameters
    batch_size = 2
    seq_len = 4
    d_model = CONFIG["d_model"]
    d_ff = CONFIG["d_ff"]

    # create random input tensor
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"Input shape: {x.shape}")

    # initialize feed-forward network
    ff_network = FeedForward(d_model, d_ff)

    # forward pass
    output = ff_network(x)
    print(f"Output shape: {output.shape}")

    # check output shape matches input shape
    assert output.shape == x.shape, (
        f"Expected output shape {x.shape}, got {output.shape}"
    )

    print("Feed-Forward Network test completed!\n")

    return output


def test_encoder_layer():
    print("Testing Encoder Layer...")

    # parameters
    batch_size = 2
    seq_len = 4
    d_model = CONFIG["d_model"]
    num_heads = CONFIG["n_heads"]
    d_ff = CONFIG["d_ff"]

    # create random input tensor
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"Input shape: {x.shape}")

    # create a sample mask
    mask = torch.ones(batch_size, 1, 1, seq_len)
    mask[:, :, :, 0] = 0  # First token is masked

    # initialize encoder layer
    encoder_layer = EncoderLayer(d_model, num_heads, d_ff)

    # forward pass
    output, attention_weights = encoder_layer(x, mask)
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")

    # check output shape matches input shape
    assert output.shape == x.shape, (
        f"Expected output shape {x.shape}, got {output.shape}"
    )

    # visualize attention weights
    plt.figure(figsize=(8, 6))
    plt.imshow(attention_weights[0, 0].detach().numpy(), cmap="viridis")
    plt.colorbar()
    plt.xlabel("Key positions")
    plt.ylabel("Query positions")
    plt.title("Encoder Layer Self-Attention Weights")
    output_path = os.path.join(output_dir, "encoder_layer_attention.png")
    plt.savefig(output_path)
    print(f"Encoder Layer attention weights visualization saved to {output_path}")

    print("Encoder Layer test completed!\n")

    return output, attention_weights


def test_encoder():
    print("Testing Encoder Stack...")

    # parameters
    batch_size = 2
    seq_len = 4
    d_model = CONFIG["d_model"]
    num_heads = CONFIG["n_heads"]
    num_layers = CONFIG["n_layers"]
    d_ff = CONFIG["d_ff"]

    # create random input tensor
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"Input shape: {x.shape}")

    # initialize encoder
    encoder = Encoder(num_layers, d_model, num_heads, d_ff)

    # forward pass
    output, attention_weights_list = encoder(x)
    print(f"Output shape: {output.shape}")
    print(f"Number of attention weight tensors: {len(attention_weights_list)}")

    # check output shape matches input shape
    assert output.shape == x.shape, (
        f"Expected output shape {x.shape}, got {output.shape}"
    )

    # verify we got attention weights for each layer
    assert len(attention_weights_list) == num_layers, (
        f"Expected {num_layers} attention weight tensors, got {len(attention_weights_list)}"
    )

    # visualize attention weights from the last layer
    plt.figure(figsize=(8, 6))
    plt.imshow(attention_weights_list[-1][0, 0].detach().numpy(), cmap="viridis")
    plt.colorbar()
    plt.xlabel("Key positions")
    plt.ylabel("Query positions")
    plt.title("Encoder Final Layer Self-Attention Weights")
    output_path = os.path.join(output_dir, "encoder_final_layer_attention.png")
    plt.savefig(output_path)
    print(f"Encoder final layer attention weights visualization saved to {output_path}")

    print("Encoder Stack test completed!")

    return output, attention_weights_list


if __name__ == "__main__":
    # test the encoder components
    test_layer_norm()
    test_feed_forward()
    test_encoder_layer()
    test_encoder()

    print("\nAll encoder component tests completed successfully!")
