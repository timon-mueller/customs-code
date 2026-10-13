import torch
import torch.nn as nn
from transformers import AutoModel

class RobertaMappingModel(nn.Module):
    def __init__(self, backbone_name, num_classes):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        self.classifier = nn.Linear(self.backbone.config.hidden_size, num_classes)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        sentence_embeddings = self.mean_pooling(outputs, attention_mask)
        logits = self.classifier(sentence_embeddings)
        return logits