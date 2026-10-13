import json
import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class CustomsTariffDataset(Dataset):
    def __init__(self, data_dir, chapters_file):
        """
        Initializes the dataset by loading sample JSON files and the chapter descriptions.
        """
        self.data = []
        self.labels = []
        
        self.chapter_to_idx = {}
        self.idx_to_chapter = {}
        self.chapter_descriptions = {}

        # 1. Load the official chapter descriptions FIRST
        if not os.path.exists(chapters_file):
            raise FileNotFoundError(f"Chapter mapping file not found: {chapters_file}")
            
        with open(chapters_file, 'r', encoding='utf-8') as f:
            chapters_data = json.load(f)
            for item in chapters_data:
                # Store mapping of chapter code to official description
                self.chapter_descriptions[str(item['chapter'])] = item['description']

        # 2. Load the training data
        json_files = glob.glob(os.path.join(data_dir, '*.json'))
        
        if not json_files:
            print(f"Warning: No JSON files found in {data_dir}")

        for file_path in json_files:
            # Prevent loading the chapters file as training data if it's in the same folder
            if os.path.abspath(file_path) == os.path.abspath(chapters_file):
                continue
                
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    file_data = json.load(f)
                    if isinstance(file_data, dict):
                        file_data = [file_data]
                        
                    for item in file_data:
                        chapter = item.get('chapter')
                        desc = item.get('description')
                        
                        # Only add to dataset if we have a valid chapter AND we know its official description
                        if chapter is not None and desc is not None:
                            chapter_str = str(chapter)
                            if chapter_str in self.chapter_descriptions:
                                self.data.append(desc)
                                self.labels.append(chapter_str)
                            else:
                                print(f"Warning: Chapter {chapter_str} not found in chapter mapping. Skipping.")
                except json.JSONDecodeError:
                    print(f"Error reading {file_path}. Ensure it is valid JSON.")

        # Create numerical indexes for the labels
        unique_chapters = sorted(list(set(self.labels)))
        self.chapter_to_idx = {chapter: idx for idx, chapter in enumerate(unique_chapters)}
        self.idx_to_chapter = {idx: chapter for chapter, idx in self.chapter_to_idx.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # The sample description (e.g., "Frozen chicken breast")
        sample_text = self.data[idx]
        
        # The official chapter text (e.g., "MEAT AND EDIBLE MEAT OFFAL")
        chapter_str = self.labels[idx]
        official_chapter_text = self.chapter_descriptions[chapter_str]
        
        # We also return the label index, which is helpful for evaluating accuracy later
        label_idx = self.chapter_to_idx[chapter_str]
        
        return sample_text, official_chapter_text, torch.tensor(label_idx, dtype=torch.long)


def get_customs_dataloaders(data_dir, chapters_file, batch_size=16, split_ratios=(0.8, 0.1, 0.1), seed=42):
    """
    Loads the dataset and splits it into Train, Validation, and Test loaders.
    """
    dataset = CustomsTariffDataset(data_dir, chapters_file)
    total_size = len(dataset)
    
    if total_size == 0:
        return None, None, None, dataset

    train_ratio, val_ratio, test_ratio = split_ratios
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    print(f"Splitting data (Total: {total_size}) -> Train: {train_size}, Val: {val_size}, Test: {test_size}")

    generator = torch.Generator().manual_seed(seed)

    train_set, val_set, test_set = random_split(
        dataset, 
        [train_size, val_size, test_size], 
        generator=generator
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, dataset