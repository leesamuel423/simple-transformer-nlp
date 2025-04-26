# simple transformer
A simplified implementation the transformer architecture from the paper [Attention is All You Need](https://arxiv.org/pdf/1706.03762).


## Notes

- Traditional NLP models mainly used RNNs or LSTMs, but had following issues:
    1. They processed text sequentially (word after word), making them slow to train
    2. They struggled with "long-range dependencies" (connecting words that are far apart in a sentence)

- Transformer architecture solves these problems by replacing recurrence with "attention", allowing model to look at words in a sentence simultaneously and determine which are most important for understanding each specific word.

- Word embeddings: before processing, each word converted to vector of numbers that places similar words closer together in mathematical space.
- Attention mechanism: assign importance scores to all words in sentence based on how relevant they are to understanding the current word.
- Self-attention: model analyzes how each word relates to every other word in the same sentence.
```
How self attention works
1. For each word, create 3 vectors:
    - (Q)uery: What this word is asking about
    - (K)ey: what this word offers to other words
    - (V)alue: actual info this word contains
2. To calculate attention:
    - Take dot product of a query with all keys to get "scores"
    - Scale scores and apply softmax to get "weights"
    - Multiply each value vector by its corresponding weight
    - Sum these weighted values to get output
```
- Multi-head attention: instead of having one way of attending to sentence, model uses multiple "attention heads" that can each focus on diff types of relationships between words.
- Positional encoding: since model processes all words at once, we need to tell it where each word appears in the sentence. Done by sine & cosine functions of different frequencies
- Layer normalization: helps keep values flowing through network in a reasonable range, making training more stable
- Residual connections: allows info to flow directly from earlier layer to later layers, helping with training deep networks

### Transformer architecture simplified
- Encoder processes input text
    - Input: word embeddings + positional encodings
    - Process:
        1. Multi-head self-attention looks at relationships between all words
        2. Feed-forward neural network processes each position
        3. Layer normalization and residual connections maintain info flow

- Decoder generates output text
    - Similar to encoder, but adds:
        1. Masked attention (prevent "cheating" by looking at future words)
        2. Attention to encoder output (connects input to output)

