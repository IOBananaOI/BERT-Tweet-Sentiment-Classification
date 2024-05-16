import torch 
import torch.nn.functional as F

from torch import nn

from transformers import BertModel


class BertTweetClf(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        
        self.clf = nn.Sequential(
            nn.Linear(768, 768),
            nn.Dropout(0.2),
            nn.Linear(768, 5)
        )
        
    def forward(self, x, attn_mask):
        return self.clf(self.bert(x, attention_mask=attn_mask).pooler_output)
        