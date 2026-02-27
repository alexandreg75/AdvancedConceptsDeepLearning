import time
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from captum.attr import IntegratedGradients, NoiseTunnel


class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x).logits


print("--- INTEGRATED GRADIENTS + SMOOTHGRAD ---")

# 1. Chargement de l'image et du modèle
image_path = sys.argv[1] if len(sys.argv) > 1 else "normal_1.jpeg"
print(f"Analyse fine au pixel sur : {image_path}")
image = Image.open(image_path).convert("RGB")

model_name = "Aunsiels/resnet-pneumonia-detection"
processor = AutoImageProcessor.from_pretrained(model_name)
hf_model = AutoModelForImageClassification.from_pretrained(model_name)

wrapped_model = ModelWrapper(hf_model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wrapped_model.to(device)
wrapped_model.eval()

inputs = processor(images=image, return_tensors="pt")
input_tensor = inputs["pixel_values"].to(device)
input_tensor.requires_grad_(True)

# Warm-up (Cold Start fix)
_ = wrapped_model(input_tensor)

# Inférence propre + timing
start_infer = time.time()
logits = wrapped_model(input_tensor)
predicted_class_idx = logits.argmax(-1).item()
end_infer = time.time()

print(f"Temps d'inférence : {end_infer - start_infer:.4f}s")
print(f"Classe prédite : {hf_model.config.id2label[predicted_class_idx]}")

# 2. Integrated Gradients
ig = IntegratedGradients(wrapped_model)

# Baseline neutre : image noire (même forme que l'input)
baseline = torch.zeros_like(input_tensor)

start_ig = time.time()
attributions_ig = ig.attribute(
    input_tensor,
    baselines=baseline,
    target=predicted_class_idx,
    n_steps=50,
    internal_batch_size=2
)
end_ig = time.time()

# 3. SmoothGrad (via NoiseTunnel)
noise_tunnel = NoiseTunnel(ig)

# IMPORTANT :
# - nt_samples trop grand => très long
# - internal_batch_size doit être <= nt_samples, sinon Captum le remonte
NT_SAMPLES = 20          # mets 10 si tu veux encore plus rapide
INTERNAL_BS = 2          # OK car <= NT_SAMPLES
STDEVS = 0.1

start_sg = time.time()
attributions_sg = noise_tunnel.attribute(
    input_tensor,
    nt_samples=NT_SAMPLES,
    nt_type="smoothgrad",
    target=predicted_class_idx,
    stdevs=STDEVS,
    internal_batch_size=INTERNAL_BS
)
end_sg = time.time()

print(f"Temps IG pur : {end_ig - start_ig:.4f}s")
print(f"Temps SmoothGrad (IG x {NT_SAMPLES}) : {end_sg - start_sg:.4f}s")

# 4. Visualisation (Valeur absolue + seuillage)
# Somme sur les canaux RGB -> (H, W)
attr_ig_vis = np.sum(np.abs(attributions_ig.squeeze().detach().cpu().numpy()), axis=0)
attr_sg_vis = np.sum(np.abs(attributions_sg.squeeze().detach().cpu().numpy()), axis=0)

# Seuillage : on garde uniquement les pixels les plus importants (au-dessus du 70e centile)
threshold_ig = np.percentile(attr_ig_vis, 70)
attr_ig_vis[attr_ig_vis < threshold_ig] = 0

threshold_sg = np.percentile(attr_sg_vis, 70)
attr_sg_vis[attr_sg_vis < threshold_sg] = 0

# Normalisation pour l'affichage
vmax_ig = np.max(attr_ig_vis) if np.max(attr_ig_vis) > 0 else 1.0
vmax_sg = np.max(attr_sg_vis) if np.max(attr_sg_vis) > 0 else 1.0

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Image originale
axes[0].imshow(image.resize(input_tensor.shape[2:][::-1]))
axes[0].set_title("Image Originale")
axes[0].axis("off")

# IG
axes[1].imshow(image.resize(input_tensor.shape[2:][::-1]), alpha=0.6)
axes[1].imshow(attr_ig_vis, cmap="hot", alpha=0.6, vmin=0, vmax=vmax_ig)
axes[1].set_title("Integrated Gradients (Seuillé)")
axes[1].axis("off")

# SmoothGrad
axes[2].imshow(image.resize(input_tensor.shape[2:][::-1]), alpha=0.6)
axes[2].imshow(attr_sg_vis, cmap="hot", alpha=0.6, vmin=0, vmax=vmax_sg)
axes[2].set_title(f"SmoothGrad (IG x {NT_SAMPLES}, Seuillé)")
axes[2].axis("off")

output_filename = f"ig_smooth_{image_path.split('.')[0]}.png"
plt.savefig(output_filename, bbox_inches="tight")
print(f"Visualisation sauvegardée dans {output_filename}")