# Customs Code

A machine learning system for automated customs tariff classification using dual encoder models and semantic search.

## Overview

This project trains neural models to classify product descriptions into customs tariff chapters. It uses a **dual encoder architecture** with multilingual transformers (XLM-RoBERTa) to learn semantic similarities between product samples and official tariff chapter descriptions.

### Key Features

- **Dual Encoder Model**: Uses contrastive learning to match product descriptions with tariff chapters
- **Multilingual Support**: Built on XLM-RoBERTa for multilingual customs data
- **Semantic Search**: Fast retrieval of relevant chapters using sentence transformers
- **Configuration-Driven**: All hyperparameters in `config/config.yaml`
- **Training Tracking**: Integrates with TensorBoard for monitoring

## Project Structure

```
.
├── models/                 # Model architectures
│   ├── dual_encoder.py    # Dual encoder implementation with contrastive learning
│   ├── semantic_index.py  # Semantic indexing for fast retrieval
│   ├── factory.py         # Model factory for instantiation
│   └── roberta_mapping.py # RoBERTa-based mapping models
├── tools/                  # Training and inference scripts
│   ├── train.py           # Training pipeline
│   ├── inference.py       # Model inference and evaluation
│   └── semantic_search.py # Semantic search utilities
├── utils/                  # Utilities
│   ├── config_loader.py   # Load configuration from YAML
│   ├── dataloader.py      # Dataset loading and splitting
│   └── model_saver.py     # Model checkpointing
├── dataset/                # Data files
│   ├── example_data/      # Sample product descriptions
│   └── zolltarif_chapters.json # Official tariff chapters
├── config/                 # Configuration
│   └── config.yaml        # Hyperparameters and settings
├── webscraping/           # Data collection
│   ├── generate.py        # Generate example data
│   └── webscrape.py       # Web scraping utilities
└── requirements.txt       # Python dependencies
```

## Installation

### Prerequisites

- Python 3.12
- Conda (recommended) or pip

### Setup

1. **Clone the repository**

```bash
git clone https://gitlab.gwdg.de/customs_project/customs-code.git
cd customs-code
```

2. **Create environment from requirements**

```bash
# Option A: Using conda (recommended)
conda create --name customs --file requirements.txt

# Option B: Using pip
pip install -r requirements.txt
```

3. **Activate environment**

```bash
conda activate customs
```

## Configuration

Edit `config/config.yaml` to adjust settings:

```yaml
model:
  name: "dual_encoder"
  pretrained_backbone: "xlm-roberta-base"
  freeze_embeddings: true
  freeze_layers: 6  # 0 (none) to 12 (all)

training:
  batch_size: 64
  epochs: 25
  learning_rate: 0.0001
  seed: 42
  split_ratios: [0.8, 0.1, 0.1]  # Train, Val, Test

semantic_search:
  model_name: "all-MiniLM-L6-v2"
  top_k: 10
```

## Usage

### Training

Start training with the configuration in `config/config.yaml`:

```bash
python -m tools.train
```

Monitor training with TensorBoard:

```bash
tensorboard --logdir models/checkpoints/dual_encoder_v1
```

### Inference

Classify a single product description:

```bash
python tools/inference.py \
  --model_name dual_encoder_v1 \
  --epoch 20 \
  --mode single \
  --text "apple"
```

Evaluate on test split:

```bash
python tools/inference.py \
  --model_name dual_encoder_freeze_6_2026-06-26_16-54 \
  --epoch 21 \
  --mode test_split
```

### Semantic Search

Find relevant tariff chapters for a product:

```python
from tools.semantic_search import SemanticSearcher

searcher = SemanticSearcher(model_name="all-MiniLM-L6-v2")
results = searcher.search("fresh apple juice", top_k=10)
```

## Model Architecture

### Dual Encoder

The dual encoder model learns to align product descriptions with tariff chapters through contrastive learning:

1. **Shared Backbone**: XLM-RoBERTa transformer encodes both product and chapter text
2. **Mean Pooling**: Average token embeddings to create sentence vectors
3. **L2 Normalization**: Normalize embeddings for cosine similarity
4. **InfoNCE Loss**: In-batch negatives for efficient contrastive training

### Training Process

- Uses mean pooling to create fixed-size embeddings
- Applies cosine similarity with temperature scaling
- Implements cross-entropy loss over in-batch negatives
- Supports layer freezing for transfer learning

## Data Format

Product samples should have the corresponding tariff chapter labels. The system expects:

```json
{
  "description": "product description text",
  "chapter": "tariff chapter number"
}
```

Chapter descriptions are loaded from `dataset/zolltarif_chapters.json`.

## Output

Models are saved to `models/checkpoints/` with structure:

```
models/checkpoints/{model_name}_{timestamp}/
├── epoch_*.pth      # Checkpoint at each epoch
├── config.yaml      # Training configuration
└── idx_to_chapter.json # Chapter index mapping
```

## Development

### Project Status

Currently implementing and evaluating dual encoder approaches for customs tariff classification. Recent improvements focus on semantic indexing and multilingual capabilities.

### Next Steps

- Evaluate alternative loss functions (e.g., triplet loss)
- Benchmark semantic search performance
- Add data augmentation strategies
- Integrate web UI for predictions

## Dependencies

Key libraries:

- **PyTorch 2.5**: Deep learning framework
- **Transformers 5.12**: Pre-trained language models
- **Sentence-Transformers**: Semantic search models
- **TensorBoard**: Training visualization
- **TorchVision/BeautifulSoup**: Data utilities

See `requirements.txt` for complete list.

## License

This project is part of the Customs classification research initiative.
