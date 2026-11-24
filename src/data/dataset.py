import os
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from .preprocessing import clean_text, get_image_transform


class MultimodalDataset(Dataset):
    """TSV file + folder of <id>.jpg images.

    split: None (use all rows) or 'train' | 'val' | 'test'
           -> stratified 70/15/15 split of the rows that have an image.
    """

    def __init__(self, tsv_path, image_dir, train=False, tokenizer_name="bert-base-uncased",
                 max_len=64, label_col="2_way_label", text_col="clean_title", id_col="id",
                 limit=None, split=None, split_seed=42):
        df = pd.read_csv(tsv_path, sep="\t", usecols=lambda c: c in {id_col, text_col, label_col})
        df = df.dropna(subset=[text_col, label_col])
        img_dir = Path(image_dir)
        # one directory listing instead of one exists() call per row (much faster)
        available = {f[:-4] for f in os.listdir(img_dir) if f.endswith(".jpg")}
        df = df[df[id_col].astype(str).isin(available)].reset_index(drop=True)

        if split is not None:
            tr, rest = train_test_split(df, test_size=0.30, stratify=df[label_col],
                                        random_state=split_seed)
            va, te = train_test_split(rest, test_size=0.50, stratify=rest[label_col],
                                      random_state=split_seed)
            df = {"train": tr, "val": va, "test": te}[split]

        if limit:
            df = df.head(limit)
        df = df.reset_index(drop=True)
        df["img_path"] = df[id_col].astype(str).map(lambda i: str(img_dir / f"{i}.jpg"))
        print(f"[dataset] {Path(tsv_path).name} ({split or 'all'}): {len(df)} samples with image")
        self.df = df
        self.text_col, self.label_col = text_col, label_col
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.transform = get_image_transform(train=train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        enc = self.tokenizer(
            clean_text(row[self.text_col]),
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        try:
            img = Image.open(row["img_path"]).convert("RGB")
        except Exception:  # corrupted image -> neutral placeholder
            img = Image.new("RGB", (224, 224))
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "image": self.transform(img),
            "label": torch.tensor(int(row[self.label_col]), dtype=torch.long),
        }
