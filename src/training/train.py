"""Usage (from repo root):
python -m src.training.train --mode multimodal --fusion gated --epochs 3
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import MultimodalDataset
from src.evaluation.metrics import evaluate
from src.models.classifier import FakeShieldModel


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--train_tsv", default="data/raw/multimodal_train.tsv")
    p.add_argument("--val_tsv", default="data/raw/multimodal_validate.tsv")
    p.add_argument("--test_tsv", default="data/raw/multimodal_test_public.tsv")
    p.add_argument("--single_tsv", default=None,
                   help="use ONE tsv, split 70/15/15 (stratified) over rows that have an image")
    p.add_argument("--image_dir", default="data/raw/images")
    p.add_argument("--mode", default="multimodal", choices=["text", "image", "multimodal"])
    p.add_argument("--fusion", default="concat", choices=["concat", "gated", "attention"])
    p.add_argument("--label_col", default="2_way_label")
    p.add_argument("--num_classes", type=int, default=2)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr_pretrained", type=float, default=2e-5)
    p.add_argument("--lr_head", type=float, default=1e-3)
    p.add_argument("--max_len", type=int, default=64)
    p.add_argument("--limit", type=int, default=None, help="subsample for quick tests")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42, help="training seed (init, shuffling)")
    p.add_argument("--split_seed", type=int, default=42, help="seed of the 70/15/15 split")
    return p.parse_args()


def main():
    a = parse()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    kw = dict(image_dir=a.image_dir, max_len=a.max_len, label_col=a.label_col, limit=a.limit)
    if a.single_tsv:
        kw["split_seed"] = a.split_seed
        train_ds = MultimodalDataset(a.single_tsv, train=True, split="train", **kw)
        val_ds = MultimodalDataset(a.single_tsv, train=False, split="val", **kw)
    else:
        train_ds = MultimodalDataset(a.train_tsv, train=True, **kw)
        val_ds = MultimodalDataset(a.val_tsv, train=False, **kw)
    mk = lambda ds, sh: DataLoader(ds, batch_size=a.batch_size, shuffle=sh, num_workers=a.workers)
    train_dl, val_dl = mk(train_ds, True), mk(val_ds, False)

    model = FakeShieldModel(a.mode, a.fusion, a.num_classes).to(device)
    pre = []
    if model.text_encoder is not None:
        pre += list(model.text_encoder.bert.parameters())
    if model.image_encoder is not None:
        pre += list(model.image_encoder.backbone.parameters())
    pre_ids = {id(p) for p in pre}
    head = [p for p in model.parameters() if id(p) not in pre_ids]
    opt = torch.optim.AdamW([{"params": pre, "lr": a.lr_pretrained},
                             {"params": head, "lr": a.lr_head}], weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_fn = nn.CrossEntropyLoss()

    name = a.mode if a.mode != "multimodal" else f"multimodal_{a.fusion}"
    if a.seed != 42:
        name += f"_s{a.seed}"
    Path("models").mkdir(exist_ok=True)
    Path("reports/results").mkdir(parents=True, exist_ok=True)
    best_f1, history = -1.0, []

    for epoch in range(1, a.epochs + 1):
        model.train()
        total = 0.0
        for b in tqdm(train_dl, desc=f"Epoch {epoch}/{a.epochs}"):
            opt.zero_grad()
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(b["input_ids"].to(device), b["attention_mask"].to(device),
                               b["image"].to(device))
                loss = loss_fn(logits, b["label"].to(device))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item()
        m = evaluate(model, val_dl, device)
        m.update(epoch=epoch, train_loss=total / len(train_dl))
        history.append(m)
        print(f"[val] epoch {epoch}: loss={m['train_loss']:.4f} acc={m['accuracy']:.4f} f1={m['f1_macro']:.4f}")
        if m["f1_macro"] > best_f1:
            best_f1 = m["f1_macro"]
            torch.save({"state_dict": model.state_dict(),
                        "config": {"mode": a.mode, "fusion": a.fusion, "num_classes": a.num_classes,
                                   "max_len": a.max_len}},
                       f"models/{name}.pt")

    # final test with the best checkpoint
    ckpt = torch.load(f"models/{name}.pt", map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    if a.single_tsv:
        test_ds = MultimodalDataset(a.single_tsv, train=False, split="test", **kw)
    else:
        test_ds = MultimodalDataset(a.test_tsv, train=False, **kw)
    test_m, y_true, y_prob = evaluate(model, mk(test_ds, False), device, return_outputs=True)
    np.savez_compressed(f"reports/results/{name}_test_preds.npz", y_true=y_true, y_prob=y_prob)
    result = {"model": name, "seed": a.seed, "n_params": sum(p.numel() for p in model.parameters()),
              "val_history": history, "test": test_m}
    Path(f"reports/results/{name}.json").write_text(json.dumps(result, indent=2))
    print("TEST:", json.dumps(result["test"], indent=2))


if __name__ == "__main__":
    main()
