# simple transformer
A simplified implementation the transformer architecture from the paper [Attention is All You Need](https://arxiv.org/pdf/1706.03762).
- Only implements encoder portion of transformer
- Focus on sequence classification rather than sequence-to-sequence tasks
- Smaller model sizes for faster training

## Installation

1. Create a virtual environment:
```bash
python -m venv transformer_env
source transformer_env/bin/activate  # On macOS/Linux
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Verify PyTorch installation
```bash
python tests/test_torch.py
```

## Components
1. Attention Mechanisms
    - Scaled Dot-Product Attention: The core attention mechanism
    - Multi-Head Attention: Multiple attention heads in parallel

2. Encoder Components
    - Position-wise Feed-Forward Network: Processes each position independently
    - Layer Normalization: Normalizes inputs across features
    - Encoder Layer: Combines attention and feed-forward with residual connections
    - Encoder: Stacks multiple encoder layers

3. Complete Transformer
    - TransformerEncoder: Full encoder with embeddings and positional encoding
    - TransformerForSequenceClassification: Adds classification head for text classification

## Usage
### Testing Components
```bash
# Test attention mechanisms
python tests/test_attention.py

# Test encoder components
python tests/test_encoder.py

# Test full transformer
python tests/test_transformer.py
```

### Training
Train model on sample sentiment analysis task:
```bash
python train.py --output_dir results --num_classes 2
```

### Evaluation
Evaluate a trained model:
```bash
python evaluate.py --model_dir results --num_classes 2
```

### Visualization
Visualize attention patterns:
```bash
python visualize.py --results_dir results --text "This movie was fantastic!"
```

## Outputs
- Trained models are saved to `results/models/`
- Trained curves are saved to `results/training_curves.png`
- Attention visualizations are saved to `visualizations/attention_visualizations/`
- Test results are saved to `results/evaluation/`

## Results

### Training Results
I trained the model for 5 epochs:

![Training Curves](/results/training_curves.png)

Key observations:
* Training accuracy jumps between 38% and 50%
* Validation accuracy stays flat at 50%
* Loss values show similar uneven patterns
* Need more training time for stable results

### Test Results
Final metrics were basic but functional:
* Accuracy: 50%
* Precision: 25%
* Recall: 50%
* F1: 33.3%

Confusion matrix shows all predictions went to class 1:

![Confusion Matrix](/results/confusion_matrix.png)

### Attention Patterns
We can see how attention works on a sample sentence:

![Attention Weights](/results/attention_visualization.png)

Different attention heads focus on different word relationships:

![Attention Heads](/visualizations/attention_visualizations/attention_heads.png)

The bright yellow spots show where the model pays most attention. You can see:
* Some heads track nearby words
* Others connect start/end tokens
* Different patterns emerge in each head

### Improving Model Performance
The current results (50% accuracy, 25% precision) aren't great. All predictions went to class 1, suggesting the model isn't learning much. Some ways to fix this:

#### Dataset Issues
* Too little training data - this implementation uses minimal examples because I was being lazy lol.
* Imbalanced classes - model just predicts majority class
* Need more diverse examples covering both classes

#### Model Improvements
* Train longer - 5 epochs isn't enough for convergence
* Adjust learning rate - start higher then decay
* Add dropout (0.1-0.2) to prevent overfitting
* Use larger model - more layers, wider feedforward layers
* Try different optimizers like AdamW

#### Training Process
* Implement early stopping based on validation loss
* Add learning rate scheduler
* Use gradient clipping to stabilize training
* Balance the dataset with weighted sampling

The main issue is likely the minimal dataset size used in this demo. Transformers typically need thousands of examples to learn effectively.

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

## Acknowledgements
This implementation is based on the paper:

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention is all you need. In Advances in Neural Information Processing Systems (pp. 5998-6008).
