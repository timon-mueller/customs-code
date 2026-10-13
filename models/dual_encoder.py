import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class DualEncoderModel(nn.Module):
    def __init__(self, backbone_name="microsoft/deberta-v3-base", temperature=0.05):
        super().__init__()
        # We use a single shared backbone to encode BOTH the sample and the chapter
        self.backbone = AutoModel.from_pretrained(backbone_name)
        
        # Temperature scales the cosine similarities to make the CrossEntropy loss more confident.
        # 0.05 is a standard default for text-based contrastive learning.
        self.temperature = temperature

    def mean_pooling(self, model_output, attention_mask):
        """Averages the token vectors to create a single sentence vector."""
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode(self, input_ids, attention_mask):
        """Passes text through the transformer and normalizes the resulting vector."""
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = self.mean_pooling(outputs, attention_mask)
        
        # L2 Normalization is CRITICAL for Cosine Similarity. 
        # It forces all vectors to have a length of exactly 1.0.
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

    # def forward(self, sample_input_ids, sample_attention_mask, chapter_input_ids, chapter_attention_mask):
    #     """
    #     During training, we pass BOTH the batch of samples and the batch of 
    #     corresponding official chapter descriptions.
    #     """
    #     # 1. Encode both sets of text independently
    #     sample_embeddings = self.encode(sample_input_ids, sample_attention_mask)
    #     chapter_embeddings = self.encode(chapter_input_ids, chapter_attention_mask)

    #     # 2. Calculate the Cosine Similarity Matrix (Batch Size x Batch Size)
    #     # Using matrix multiplication (dot product of normalized vectors = cosine similarity)
    #     # We divide by the temperature to sharpen the scores for CrossEntropy
    #     similarity_matrix = torch.matmul(sample_embeddings, chapter_embeddings.T) / self.temperature

    #     # 3. Calculate InfoNCE Loss (In-Batch Negatives)
    #     # The correct matches are exactly on the diagonal (0 matches 0, 1 matches 1, etc.)
    #     batch_size = sample_embeddings.size(0)
        
    #     # Create target labels: [0, 1, 2, ..., batch_size - 1]
    #     targets = torch.arange(batch_size, device=sample_embeddings.device)
        
    #     # Cross Entropy pushes the diagonal values toward 1.0 and all others toward 0.0
    #     # We calculate loss in both directions (Sample -> Chapter AND Chapter -> Sample)
    #     loss_sample_to_chapter = F.cross_entropy(similarity_matrix, targets)
    #     loss_chapter_to_sample = F.cross_entropy(similarity_matrix.T, targets)
        
    #     # The final loss is the average of both directions
    #     total_loss = (loss_sample_to_chapter + loss_chapter_to_sample) / 2

    #     return total_loss, similarity_matrix