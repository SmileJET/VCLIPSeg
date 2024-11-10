import os
import clip
import torch


## Category name
ORGAN_NAME = ['Background', 'Left Atrial']

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# A pixel of {Category}.
text_inputs = torch.cat([clip.tokenize(f'A pixel of a {item}.') for item in ORGAN_NAME]).to(device)

# Calculate text embedding features
with torch.no_grad():
    text_features = model.encode_text(text_inputs)
    text_features = text_features.float()
    print(text_features.shape, text_features.dtype)
    torch.save(text_features, 'la_txt_encoding.pth')
