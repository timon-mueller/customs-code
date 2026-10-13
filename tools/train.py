import os
import sys
import json
import yaml
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from utils.config_loader import load_config
from utils.dataloader import get_customs_dataloaders
from models.factory import get_model
from utils.model_saver import ModelSaver

def main():
    # we try to load all the parameters from yaml to keep everything configurable in one place
    config = load_config()

    # stuff for model loading - ignore it
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # load the chapter descriptions
    data_dir = config['dataset']['example_data_dir']
    chapters_file = os.path.join(project_root, 'dataset', 'zolltarif_chapters.json') 
    
    # load the hyperparameter from yaml
    batch_size = config['training']['batch_size']
    epochs = config['training']['epochs']
    lr = config['training']['learning_rate']

    # initialize the dataloaders - they provide our train data in a usefull shape
    train_loader, val_loader, test_loader, dataset = get_customs_dataloaders(
        data_dir=data_dir, 
        chapters_file=chapters_file,
        batch_size=batch_size, 
        split_ratios=config['training']['split_ratios'], 
        seed=config['training']['seed']
    )
    
    num_classes = len(dataset.idx_to_chapter)
    chapter_texts = [dataset.chapter_descriptions[str(dataset.idx_to_chapter[i])] for i in range(num_classes)]

    # logic for saving the model weigths to load them later
    saver = ModelSaver(project_root, config, dataset.idx_to_chapter)
    # we create all models in the factory.py class 
    model = get_model(config, num_classes=num_classes) 
    # part of the transformers package - we just have to pass it the name of our backbone and we get the fitting Tokenizer
    #.it's job is to transform out text into tensor
    tokenizer = AutoTokenizer.from_pretrained(config['model']['pretrained_backbone'])

    # we can't train the whole Transformer for 2 reasons:
    # 1. we should't overwrite all the weights -> catastrophic forgetting
    # 2. it blows up my mac, we need about 32gb of unified memory for xlm-roberta-base
    freeze_embeddings = config['model'].get('freeze_embeddings', True)
    freeze_layers = config['model'].get('freeze_layers', 6)

    if freeze_embeddings:
        print(f"Freezing embeddings: {freeze_embeddings}")
        print(f"Freezing first {freeze_layers} transformer layers...")
        for param in model.backbone.embeddings.parameters():
            param.requires_grad = False

    if freeze_layers > 0:
        # Dynamically slice the layers based on the config integer
        for layer in model.backbone.encoder.layer[:freeze_layers]:
            for param in layer.parameters():
                param.requires_grad = False

    # even though the optimizer is smart enought to only consider params with grad:
    # it is a good practice to give him only the parameter he needs to consider
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    
    # we are doing a 96-way multiple choice -> standard Cross Entropy loss
    criterion = torch.nn.CrossEntropyLoss()

    # this omptimizes the training speed, you should use cuda or mps cause cpu is very slow
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    print(f"Training running on device: {device}\n")

    # we need to pass the chapter describtions in the encoder over and over, so we can save time when saving them
    global_chapter_inputs = tokenizer(
        chapter_texts, padding=True, truncation=True, max_length=32, return_tensors="pt"
    ).to(device)

    # this one saves the best val epoch and is a good indicator which to pick in inference
    best_val_acc = 0.0  

    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * 0.1) # 10% of total steps for warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )

    ########################## Train loooooop ##########################################
    for epoch in range(1, epochs + 1):
        model.train()
        train_running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # keep your terminal clean :)
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]", leave=False)
        
        # dataloader returns sample text, official chapter text, chapter number
        for sample_texts, _, label_indices in progress_bar:
            current_batch_size = len(sample_texts)
            # because "tensors to be on the same device"
            targets = label_indices.to(device)
            
            # step 1 pass the text into the tokenizer
            sample_inputs = tokenizer(
                list(sample_texts), padding=True, truncation=True, max_length=32, return_tensors="pt"
            ).to(device)
            
            # we have to reset our gradients every step, if not pytorch adds them all up
            optimizer.zero_grad()
            
            # step 2.1 encode the token with 
            sample_embeddings = model.encode(sample_inputs['input_ids'], sample_inputs['attention_mask'])
            
            # step 2.2 encode all 96 chapters, must be done every step so the chapter encoder learns
            chapter_embeddings = model.encode(global_chapter_inputs['input_ids'], global_chapter_inputs['attention_mask'])
            
            # step 3 calculate Similarity [Batch_Size x 96]
            similarity_matrix = torch.matmul(sample_embeddings, chapter_embeddings.T) / model.temperature
            
            # step 4 Standard Classification Loss!
            loss = criterion(similarity_matrix, targets)
            
            # backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            scheduler.step()

            train_running_loss += loss.item() * current_batch_size
            
            # global accuracy calculation
            predictions = torch.argmax(similarity_matrix, dim=1)
            train_correct += (predictions == targets).sum().item()
            train_total += current_batch_size
            # current loss calculation
            current_loss = train_running_loss / train_total
            current_acc = (train_correct / train_total) * 100
            progress_bar.set_postfix({'Loss': f'{current_loss:.4f}', 'GlobalAcc': f'{current_acc:.1f}%'})
            
        epoch_train_loss = train_running_loss / train_total
        epoch_train_acc = (train_correct / train_total) * 100

        ########################## Validation loooooop ##########################################
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for sample_texts, _, label_indices in val_loader:
                current_batch_size = len(sample_texts)
                targets = label_indices.to(device)
                
                sample_inputs = tokenizer(list(sample_texts), padding=True, truncation=True, max_length=32, return_tensors="pt").to(device)

                sample_embeddings = model.encode(sample_inputs['input_ids'], sample_inputs['attention_mask'])
                chapter_embeddings = model.encode(global_chapter_inputs['input_ids'], global_chapter_inputs['attention_mask'])
                
                similarity_matrix = torch.matmul(sample_embeddings, chapter_embeddings.T) / model.temperature
                loss = criterion(similarity_matrix, targets)

                val_running_loss += loss.item() * current_batch_size
                predictions = torch.argmax(similarity_matrix, dim=1)
                val_correct += (predictions == targets).sum().item()
                val_total += current_batch_size

        epoch_val_loss = val_running_loss / val_total
        epoch_val_acc = (val_correct / val_total) * 100

        # epoch summary and tensorboard logging
        print(f"Epoch {epoch}/{epochs} Complete:")
        print(f"  Train -> Loss: {epoch_train_loss:.4f} | Global Acc: {epoch_train_acc:.2f}%")
        print(f"  Val   -> Loss: {epoch_val_loss:.4f} | Global Acc: {epoch_val_acc:.2f}%")

        current_lr = optimizer.param_groups[0]['lr']

        saver.log_metrics(epoch, epoch_train_loss, epoch_train_acc, epoch_val_loss, epoch_val_acc, current_lr)
        saver.save_epoch(model, epoch)
        
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            print(f"New best validation accuracy! ({best_val_acc:.2f}%)")
        print()

    saver.close()
    print("Training finished successfully!")
    
if __name__ == "__main__":
    main()