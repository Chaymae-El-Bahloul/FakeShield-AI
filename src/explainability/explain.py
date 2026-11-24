import numpy as np
import torch
from PIL import Image


def gradcam(model, image, input_ids=None, attention_mask=None, target=None):
    """Grad-CAM on ResNet50 layer4. Returns (heatmap in [0,1], target class)."""
    layer = model.image_encoder.backbone.layer4
    store = {}

    def fwd_hook(_m, _i, out):
        store["act"] = out
        out.register_hook(lambda g: store.__setitem__("grad", g))

    h = layer.register_forward_hook(fwd_hook)
    model.eval()
    model.zero_grad()
    image = image.clone().requires_grad_(True)
    logits = model(input_ids, attention_mask, image)
    if target is None:
        target = int(logits.argmax(1).item())
    logits[0, target].backward()
    h.remove()
    w = store["grad"].mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((w * store["act"]).sum(1)).squeeze(0).detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam, target


def overlay_heatmap(pil_img: Image.Image, cam: np.ndarray, alpha=0.45) -> Image.Image:
    import matplotlib.cm as cm
    heat = Image.fromarray((cam * 255).astype(np.uint8)).resize(pil_img.size, Image.BILINEAR)
    colored = (cm.jet(np.array(heat) / 255.0)[..., :3] * 255).astype(np.uint8)
    base = np.array(pil_img.convert("RGB"))
    return Image.fromarray((base * (1 - alpha) + colored * alpha).astype(np.uint8))


def text_attributions(model, tokenizer, text, image, target, device, max_len=64, n_steps=30):
    """Integrated Gradients (Captum) on BERT embeddings. Returns (tokens, scores in [-1,1])."""
    from captum.attr import LayerIntegratedGradients

    model.eval()
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len).to(device)

    def forward(input_ids, attention_mask):
        return torch.softmax(model(input_ids, attention_mask, image), dim=-1)

    lig = LayerIntegratedGradients(forward, model.text_encoder.bert.embeddings)
    baseline = torch.full_like(enc["input_ids"], tokenizer.pad_token_id)
    attr = lig.attribute(enc["input_ids"], baselines=baseline,
                         additional_forward_args=(enc["attention_mask"],),
                         target=target, n_steps=n_steps)
    scores = attr.sum(-1).squeeze(0).detach().cpu()
    scores = (scores / (scores.abs().max() + 1e-8)).tolist()
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
    return tokens, scores
