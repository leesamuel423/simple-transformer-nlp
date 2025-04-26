import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ScaledDotProductAttention(nn.Module):
    """
    scaled dot-product attention mechanism calculates attention weights by
    taking dot products of queries and keys, scaling by sqrt(d_k) and
    applying softmax
    """

    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, query, key, value, mask=None):
        """
        Forward pass for scaled dot-product attention

        Args:
            query: query tensor [batch_size, num_heads, seq_len, d_k]
            key: key tensor [batch_size, num_heads, seq_len, d_k]
            value: value tensor [batch_size, num_heads, seq_len, d_v]
            mask: optional mask to prevent attention to certain positions
                [batch_size, 1, 1, seq_len]

        Returns:
            attention_output: weighted sum of values based on attention scores
            attention_weights: attention weight distribution
        """
        # get dimensions
        d_k = query.shape[-1]

        # calculate attention scores
        # (batch_size, num_heads, seq_len, d_k) x (batch_size, num_heads, d_k, seq_len)
        # -> (batch_size, num_heads, seq_len, seq_len)
        attention_scores = torch.matmul(query, key.transpose(-2, -1))

        # scale scores by sqrt(d_k)
        attention_scores = attention_scores / math.sqrt(d_k)

        # apply mask if provided
        # used in decoder to prevent attending to future tokens
        if mask is not None:
            # Set masked positions to negative infinity so they evaluate to 0 after softmax
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        # ppply softmax to get attention weights (probabilities sum to 1)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # multiply weights by values to get the final output
        # (batch_size, num_heads, seq_len, seq_len) x (batch_size, num_heads, seq_len, d_v)
        # -> (batch_size, num_heads, seq_len, d_v)
        attention_output = torch.matmul(attention_weights, value)

        return attention_output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    multi-head attention runs multiple attention mechanisms in parallel and
    combines their outputs.
    """

    def __init__(self, d_model, num_heads):
        """
        initialize Multi-Head Attention

        Args:
            d_model: model dimension (embedding size)
            num_heads: # of attention heads
        """
        super(MultiHeadAttention, self).__init__()

        # ensure d_model is divisible by num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # dimension of each head

        # linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # output projection
        self.W_o = nn.Linear(d_model, d_model)

        # scaled dot-product attention
        self.attention = ScaledDotProductAttention()

    def split_heads(self, x, batch_size):
        """
        split the last dimension into (num_heads, d_k) and reshape to
        (batch_size, num_heads, seq_len, d_k)

        Args:
            x: input tensor [batch_size, seq_len, d_model]
            batch_size: batch size

        Returns:
            reshaped tensor [batch_size, num_heads, seq_len, d_k]
        """
        # reshape to [batch_size, seq_len, num_heads, d_k]
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        # transpose to [batch_size, num_heads, seq_len, d_k]
        return x.transpose(1, 2)

    def forward(self, query, key, value, mask=None):
        """
        forward pass for multi-head attention

        Args:
            query: query tensor [batch_size, seq_len_q, d_model]
            key: key tensor [batch_size, seq_len_k, d_model]
            value: value tensor [batch_size, seq_len_v, d_model]
            mask: optional mask to prevent attention to certain positions

        Returns:
            out: output tensor [batch_size, seq_len_q, d_model]
            attention_weights: attention weights for visualization
        """
        batch_size = query.size(0)

        # linear projections and split into multiple heads
        q = self.split_heads(
            self.W_q(query), batch_size
        )  # (batch_size, num_heads, seq_len_q, d_k)
        k = self.split_heads(
            self.W_k(key), batch_size
        )  # (batch_size, num_heads, seq_len_k, d_k)
        v = self.split_heads(
            self.W_v(value), batch_size
        )  # (batch_size, num_heads, seq_len_v, d_k)

        # apply scaled dot-product attention
        attn_output, attention_weights = self.attention(q, k, v, mask)

        # reshape attention output back to original dimensions
        # transpose from (batch_size, num_heads, seq_len_q, d_k) to (batch_size, seq_len_q, num_heads, d_k)
        attn_output = attn_output.transpose(1, 2)

        # concatenate heads and pass through final linear layer
        # reshape to (batch_size, seq_len_q, d_model)
        concat_attn = attn_output.contiguous().view(batch_size, -1, self.d_model)

        # final linear projection
        out = self.W_o(concat_attn)

        return out, attention_weights
