import torch
import torch.nn as nn
from models.embeddings import TokenEmbedding, PositionalEncoding
from models.encoder import Encoder


class TransformerEncoder(nn.Module):
    """
    complete transformer encoder model
    combines token embeddings, positional encodings, and the encoder stack
    """

    def __init__(
        self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len, dropout=0.1
    ):
        """
        initialize transformer encoder

        Args:
            vocab_size: size of the vocabulary
            d_model: model dimension
            num_heads: # of attention heads
            num_layers: # of encoder layers
            d_ff: feed-forward hidden dimension
            max_seq_len: maximum sequence length
            dropout: dropout rate
        """
        super(TransformerEncoder, self).__init__()

        # token embedding layer
        self.token_embedding = TokenEmbedding(vocab_size, d_model)

        # positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # encoder stack
        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, dropout)

        # initialize parameters
        self._init_parameters()

    def _init_parameters(self):
        """
        initialize model parameters with Xavier/Glorot initialization
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_mask=None):
        """
        forward pass for the transformer encoder

        Args:
            src: source sequence tensor [batch_size, seq_len]
            src_mask: optional mask to prevent attention to certain positions

        Returns:
            encoder_output: encoded representation of the input
            attention_weights: list of attention weights from each layer
        """
        # convert token indices to embeddings
        src_embedded = self.token_embedding(src)

        # add positional encoding
        src_embedded = self.positional_encoding(src_embedded)

        # pass through encoder
        encoder_output, attention_weights = self.encoder(src_embedded, src_mask)

        return encoder_output, attention_weights


class TransformerForSequenceClassification(nn.Module):
    """
    transformer model for sequence classification tasks
    uses the TransformerEncoder and adds a classification head.
    """

    def __init__(
        self,
        vocab_size,
        d_model,
        num_heads,
        num_layers,
        d_ff,
        max_seq_len,
        num_classes,
        dropout=0.1,
    ):
        """
        initialize transformer for sequence classification

        Args:
            vocab_size: size of the vocabulary
            d_model: model dimension
            num_heads: # of attention heads
            num_layers: # of encoder layers
            d_ff: feed-forward hidden dimension
            max_seq_len: maximum sequence length
            num_classes: # of output classes
            dropout: dropout rate
        """
        super(TransformerForSequenceClassification, self).__init__()

        # transformer encoder
        self.transformer = TransformerEncoder(
            vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len, dropout
        )

        # classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, src, src_mask=None):
        """
        forward pass for sequence classification

        Args:
            src: source sequence tensor [batch_size, seq_len]
            src_mask: optional mask to prevent attention to certain positions

        Returns:
            logits: classification logits [batch_size, num_classes]
            attention_weights: list of attention weights from each layer
        """
        # get encoder output
        encoder_output, attention_weights = self.transformer(src, src_mask)

        # use the [CLS] token representation (first token) for classification
        cls_representation = encoder_output[:, 0, :]

        # compute logits
        logits = self.classifier(cls_representation)

        return logits, attention_weights
