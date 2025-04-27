import torch
import torch.nn as nn
from models.attention import MultiHeadAttention


class LayerNorm(nn.Module):
    """
    layer normalization normalizes the inputs across the features for stability
    """

    def __init__(self, features, eps=1e-6):
        """
        initialize layer normalization

        Args:
            features: # of features in the input
            eps: small value for numerical stability
        """
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        """
        forward pass for layer normalization

        Args:
            x: input tensor [..., features]

        Returns:
            normalized tensor of the same shape
        """
        # calculate mean and variance along the last dimension
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        # normalize
        x_normalized = (x - mean) / (std + self.eps)

        # scale and shift with learnable parameters
        return self.gamma * x_normalized + self.beta


class FeedForward(nn.Module):
    """
    position-wise feed-forward network
    consists of two linear transformations with a ReLU activation in between
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        initialize position-wise feed-forward network

        Args:
            d_model: model dimension (input and output)
            d_ff: hidden layer dimension
            dropout: dropout rate
        """
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        forward pass for feed-forward network

        Args:
            x: input tensor [batch_size, seq_len, d_model]

        Returns:
            output tensor [batch_size, seq_len, d_model]
        """
        # apply first linear layer and ReLU
        x = self.relu(self.linear1(x))

        # apply dropout and second linear layer
        x = self.linear2(self.dropout(x))

        return x


class EncoderLayer(nn.Module):
    """
    encoder layer consists of multi-head self-attention and a position-wise
    feed-forward network. uses residual connections and layer normalization.
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        initialize encoder layer

        Args:
            d_model: model dimension
            num_heads: # of attention heads
            d_ff: feed-forward hidden dimension
            dropout: dropout rate
        """
        super(EncoderLayer, self).__init__()

        # self-attention with multi-head attention
        self.self_attn = MultiHeadAttention(d_model, num_heads)

        # feed-forward network
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        # layer normalization
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

        # dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        forward pass for encoder layer

        Args:
            x: input tensor [batch_size, seq_len, d_model]
            mask: optional mask to prevent attention to certain positions

        Returns:
            output tensor [batch_size, seq_len, d_model]
        """
        # self-attention with residual connection and layer normalization
        attn_output, attention_weights = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # feed-forward with residual connection and layer normalization
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x, attention_weights


class Encoder(nn.Module):
    """
    transformer encoder consists of multiple stacked encoder layers.
    """

    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        """
        initialize encoder

        Args:
            num_layers: # of encoder layers
            d_model: model dimension
            num_heads: # of attention heads
            d_ff: feed-forward hidden dimension
            dropout: dropout rate
        """
        super(Encoder, self).__init__()

        # stack of encoder layers
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        # final layer normalization
        self.norm = LayerNorm(d_model)

    def forward(self, x, mask=None):
        """
        forward pass for encoder

        Args:
            x: input tensor [batch_size, seq_len, d_model]
            mask: optional mask to prevent attention to certain positions

        Returns:
            Output tensor [batch_size, seq_len, d_model]
            attention_weights: list of attention weights from each layer
        """
        # list to store attention weights from each layer
        attention_weights = []

        # pass input through each encoder layer
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            attention_weights.append(attn_weights)

        # apply final layer normalization
        x = self.norm(x)

        return x, attention_weights
