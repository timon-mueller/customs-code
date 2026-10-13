# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Customs Code** is a machine learning system for automated customs tariff classification. It uses dual encoder models with XLM-RoBERTa transformers to match product descriptions against tariff chapter categories through contrastive learning and semantic search.

## Development Commands

### Environment Setup
```bash
# Create conda environment from requirements.txt
conda create --name customs --file requirements.txt

# Activate environment
conda activate customs
```

### Training
```bash
# Run training with config from config/config.yaml
python -m tools.train

# Monitor training with TensorBoard (in parallel terminal)
tensorboard --logdir models/checkpoints/{model_name}
```

### Inference & Evaluation
```bash
# Single prediction
python tools/inference.py \
  --model_name dual_encoder_v1 \
  --epoch 20 \
  --mode single \
  --text "product description"

# Evaluate on test split
python tools/inference.py \
  --model_name {model_checkpoint} \
  --epoch {epoch_num} \
  --mode test_split
```

### Semantic Search
```python
from tools.semantic_search import SemanticSearcher
searcher = SemanticSearcher(model_name="all-MiniLM-L6-v2")
results = searcher.search("product description", top_k=10)
```

## Architecture

### Core Components

**Models** (`models/`)
- `dual_encoder.py`: Dual encoder with contrastive learning. Uses mean pooling + L2 norm + cosine similarity with temperature scaling
- `semantic_index.py`: Semantic indexing for fast retrieval using sentence-transformers
- `factory.py`: Factory pattern for model instantiation based on config `model.name`
- `roberta_mapping.py`: Alternative RoBERTa-based classifier

**Training Pipeline** (`tools/train.py`)
1. Load config from `config/config.yaml`
2. Initialize data loaders (train/val/test split)
3. Instantiate model via factory based on config
4. Freeze embedding layers and transformer layers (configurable)
5. Train with AdamW optimizer + cross-entropy loss
6. Use linear warmup scheduler (10% warmup)
7. Save best model checkpoint based on validation accuracy
8. Save `idx_to_chapter.json` mapping for inference

**Utilities** (`utils/`)
- `config_loader.py`: YAML configuration loading
- `dataloader.py`: Dataset loading, splitting, and batching
- `model_saver.py`: Model checkpointing with timestamped directories

### Data Flow

1. Config YAML → config_loader → hyperparameters & paths
2. Dataset JSON files → dataloader → train/val/test batches
3. Batches → model → logits
4. Logits + labels → cross-entropy loss
5. Best validation checkpoint → models/checkpoints/{timestamp}/

### Configuration System

All hyperparameters live in `config/config.yaml`:
- `model.name`: Switch between `dual_encoder`, `roberta_mapping`
- `model.pretrained_backbone`: Base transformer (default: `xlm-roberta-base`)
- `model.freeze_embeddings`: Freeze embedding layer (True/False)
- `model.freeze_layers`: Number of transformer layers to freeze (0-12)
- `model.temperature`: Temperature for contrastive loss (dual_encoder only)
- `training.batch_size`, `epochs`, `learning_rate`, `seed`
- `training.split_ratios`: [train, val, test] proportions
- `semantic_search.model_name`: Sentence transformer for search

## Git & Version Control
- **Always ask before pushing** to any branch. Never push without explicit user approval, even if changes appear complete.
- Use conventional commit messages when creating commits.

## Communication Style
- **No emojis** in any output, including commit messages, comments, and text responses.
- Keep responses concise and technical.

## Code Practices
- Prefer editing existing files over creating new ones.
- Add comments only when the WHY is non-obvious (e.g., layer freezing rationale, device selection heuristics).
- Trust PyTorch and transformers library guarantees; focus validation at data boundaries.
- No premature abstractions—the factory pattern and config system are the primary design patterns used.
