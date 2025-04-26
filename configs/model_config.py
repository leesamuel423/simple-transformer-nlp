# Configuration for the Transformer model

CONFIG = {
    # Model parameters
    "d_model": 128,  # Embedding dimension
    "n_heads": 2,  # Number of attention heads
    "n_layers": 2,  # Number of encoder layers
    "d_ff": 256,  # Feed-forward layer dimension
    "dropout": 0.1,  # Dropout rate
    # Data parameters
    "max_seq_len": 50,  # Maximum sequence length
    "vocab_size": 10000,  # Vocabulary size
    # Training parameters
    "batch_size": 32,
    "learning_rate": 0.0001,
    "epochs": 5,
    "warmup_steps": 4000,
}
