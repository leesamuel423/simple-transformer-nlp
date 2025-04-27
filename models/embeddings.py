import torch
import torch.nn as nn
import math


class TokenEmbedding(nn.Module):
    """
    token embedding layer that converts token indices to dense vectors
    """

    def __init__(self, vocab_size, d_model):
        """
        initialize token embedding

        Args:
            vocab_size: size of the vocabulary
            d_model: dimension of the embedding
        """
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        """
        forward pass for token embedding

        Args:
            x: input tensor of token indices [batch_size, seq_len]

        Returns:
            embedded tensor [batch_size, seq_len, d_model]
        """
        # scale embeddings by sqrt(d_model) as mentioned in the paper
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    positional encoding adds positional information to token embeddings since the transformer
    model doesn't have any built-in sense of token order.
    """

    def __init__(self, d_model, max_seq_len=5000, dropout=0.1):
        """
        initialize positional encoding

        Args:
            d_model: dimension of the embedding
            max_seq_len: max sequence length
            dropout: dropout rate
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # [max_seq_len, d_model]
        pe = torch.zeros(max_seq_len, d_model)

        # [max_seq_len, 1]
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        # create a tensor for the division term in the positional formula
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)  # even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dimensions

        # add batch dimension [1, max_seq_len, d_model]
        pe = pe.unsqueeze(0)

        # register as a buffer (not a parameter) since it's fixed
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        forward pass for positional encoding

        Args:
            x: input tensor [batch_size, seq_len, d_model]

        Returns:
            x + pe[:, :seq_len]: input combined with positional encoding
        """
        # add positional encoding to input embeddings
        # the pe buffer is [1, max_seq_len, d_model]
        # we take only the positions we need: [:, :x.size(1), :]
        x = x + torch.tensor(self.pe)[:, : x.size(1), :].to(x.device)
        return self.dropout(x)
