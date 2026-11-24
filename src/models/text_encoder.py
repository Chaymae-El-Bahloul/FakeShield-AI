import torch.nn as nn
from transformers import AutoModel


class TextEncoder(nn.Module):
    """BERT encoder -> [CLS] -> projection to `out_dim`."""

    def __init__(self, name="bert-base-uncased", out_dim=512, dropout=0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(name)
        hidden = self.bert.config.hidden_size
        self.proj = nn.Sequential(nn.Linear(hidden, out_dim), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.proj(out.last_hidden_state[:, 0])
