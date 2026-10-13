import os
import sys
import json
import yaml
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from tqdm import tqdm

# Ensure project root is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.factory import get_model
from utils.dataloader import get_customs_dataloaders

def get_embedding(model, tokenizer, texts, device, max_length=128):
    inputs = tokenizer(
        list(texts), 
        padding=True, 
        truncation=True, 
        max_length=max_length, 
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        # Cleanly bypass the manual math and use your model's built-in pipeline
        embeddings = model.encode(inputs['input_ids'], inputs['attention_mask'])
        
    return embeddings

def main():
    parser = argparse.ArgumentParser(description="Run inference on a trained dual-encoder customs mapping model.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model directory")
    parser.add_argument("--epoch", type=int, required=True, help="Which epoch weights to load")
    parser.add_argument("--mode", type=str, choices=["single", "test_split"], default="single", help="Inference mode")
    parser.add_argument("--text", type=str, default="industrial robotic arm for manufacturing", help="Text to classify (single mode)")
    args = parser.parse_args()

    checkpoint_dir = os.path.join(project_root, "models", "checkpoints", args.model_name)
    
    if not os.path.exists(checkpoint_dir):
        print(f"Error: Model directory '{checkpoint_dir}' does not exist.")
        return

    with open(os.path.join(checkpoint_dir, "config.yaml"), "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    with open(os.path.join(checkpoint_dir, "class_mapping.json"), "r", encoding='utf-8') as f:
        idx_to_chapter = json.load(f)
    
    # Define the path ONCE here
    chapters_file = os.path.join(project_root, 'dataset', 'zolltarif_chapters.json')
    
    # Load and map the descriptions
    with open(chapters_file, "r", encoding='utf-8') as f:
        chapter_data_list = json.load(f)
    chapter_lookup = {item["chapter"]: item["description"] for item in chapter_data_list}

    chapter_texts = []
    for i in range(len(idx_to_chapter)):
        chapter_code = str(idx_to_chapter[str(i)])
        full_description = chapter_lookup[chapter_code]
        chapter_texts.append(full_description)

    num_classes = len(idx_to_chapter)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running inference on device: {device}")
    
    model = get_model(config, num_classes=num_classes)
    tokenizer = AutoTokenizer.from_pretrained(config['model']['pretrained_backbone'])

    weights_path = os.path.join(checkpoint_dir, f"model_weights_epoch_{args.epoch}.pt")
    if not os.path.exists(weights_path):
        print(f"Error: Weights file '{weights_path}' not found.")
        return
        
    print(f"Loading weights from: {weights_path}...\n")
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()

    # ==========================================
    # PHASE A: Compute All Chapter Embeddings
    # ==========================================
    print("Encoding official tariff chapters...")
    # Generate static database vectors for all 96 target chapters
    chapter_embeddings = get_embedding(model, tokenizer, chapter_texts, device)

    # ==========================================
    # PHASE B: Execute Inference Mode
    # ==========================================
    if args.mode == "single":
        print(f"--- MODE: SINGLE TEXT PREDICTION ---")
        sample_embedding = get_embedding(model, tokenizer, [args.text], device)
        
        # Compute cosine similarity via matrix multiplication
        similarity_matrix = torch.matmul(sample_embedding, chapter_embeddings.T)
        
        predicted_idx = torch.argmax(similarity_matrix, dim=1).item()
        probabilities = torch.softmax(similarity_matrix, dim=1)[0]
        confidence = probabilities[predicted_idx].item()

        print(f"Input Text:      '{args.text}'")
        print(f"Predicted Class: Chapter {idx_to_chapter[str(predicted_idx)]}")
        print(f"Confidence:      {confidence:.2%}")

    elif args.mode == "test_split":
        print(f"--- MODE: TEST SPLIT EVALUATION ---")
        
        data_dir = config['dataset']['example_data_dir']
        
        _, _, test_loader, _ = get_customs_dataloaders(
            data_dir=data_dir, 
            chapters_file=chapters_file,
            batch_size=1,#config['training']['batch_size'], 
            split_ratios=config['training']['split_ratios'], 
            seed=config['training']['seed']
        )
        
        test_correct = 0
        test_total = 0
        
        print("Evaluating entire test subset over global chapter bank...")
        for sample_texts, _, label_indices in tqdm(test_loader, desc="Testing"):
            current_batch_size = len(sample_texts)
            targets = label_indices.to(device)
            
            # Extract sample embeddings for the current batch
            sample_embeddings = get_embedding(model, tokenizer, sample_texts, device)
            
            # Compute similarity score matrix: Shape (batch_size, 96)
            similarity_matrix = torch.matmul(sample_embeddings, chapter_embeddings.T)
            
            predictions = torch.argmax(similarity_matrix, dim=1)
            test_correct += (predictions == targets).sum().item()
            test_total += current_batch_size
                
        final_accuracy = (test_correct / test_total) * 100
        print(f"\n--- EVALUATION COMPLETE ---")
        print(f"Processed items: {test_total}")
        print(f"Global Accuracy: {final_accuracy:.2f}%")

if __name__ == "__main__":
    main()