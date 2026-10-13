import os
import sys
from tqdm import tqdm

# Import your existing utilities
from utils.config_loader import load_config
from utils.dataloader import get_customs_dataloaders

# Import your coworker's class (assuming they named the file semantic_index.py)
from models.semantic_index import SemanticIndex

def main():
    config = load_config()
    
    print("Loading datasets...")
    # We load your exact dataloaders so the train/val splits are identical to your approach
    train_loader, val_loader, test_loader, dataset = get_customs_dataloaders(
        data_dir=config['dataset']['example_data_dir'], 
        chapters_file=os.path.join('dataset', 'zolltarif_chapters.json'),
        batch_size=config['training']['batch_size'], 
        split_ratios=config['training']['split_ratios'], 
        seed=config['training']['seed']
    )

    # 1. Prepare the training records for the coworker's index
    # We must format your PyTorch dataset into the [{"description": text, "chapter": label}] format they expect
    train_records = []
    print("Formatting training data for Semantic Index...")
    for text, _, label_idx in tqdm(train_loader.dataset):
        chapter_str = str(dataset.idx_to_chapter[label_idx.item()])
        train_records.append({"description": text, "chapter": chapter_str})

    # 2. Build the Semantic Index
    model_name = config.get("semantic_search", {}).get("model_name", "all-MiniLM-L6-v2")
    print(f"\nBuilding index using model: {model_name}")
    index = SemanticIndex(model_name)
    index.build(train_records)
    
    # Optional: Save it so you don't have to rebuild it every time
    # index.save(os.path.join("models", "semantic_index"))

    # 3. Evaluate on the Validation Set
    print("\nStarting evaluation on validation set...")
    correct_predictions = 0
    total_predictions = 0
    top_k = config.get("semantic_search", {}).get("top_k", 10)

    # Iterate through your validation dataset one by one
    for text, _, label_idx in tqdm(test_loader.dataset, desc="Evaluating"):
        true_chapter = str(dataset.idx_to_chapter[label_idx.item()])
        
        # Coworker's predict function
        predicted_chapter, confidence, _ = index.predict(text, top_k=top_k)
        
        if predicted_chapter == true_chapter:
            correct_predictions += 1
        total_predictions += 1

    # 4. Calculate and compare results
    accuracy = (correct_predictions / total_predictions) * 100
    print("\n" + "="*40)
    print("COWORKER MODEL EVALUATION RESULTS")
    print("="*40)
    print(f"Total Test Samples : {total_predictions}")
    print(f"Correct Predictions      : {correct_predictions}")
    print(f"Global Accuracy          : {accuracy:.2f}%")
    print("="*40)

if __name__ == "__main__":
    main()