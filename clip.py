from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

# Load CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Process images and texts
images = [Image.open("path/to/image.jpg") for image in image_paths]
texts = ["A 3D model of a car", "A 3D model of an apple", ...]
inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)

# Get embeddings
outputs = model(**inputs)
image_embeddings = outputs.image_embeds
text_embeddings = outputs.text_embeds

# Compute cosine similarities
cosine_similarities = torch.nn.functional.cosine_similarity(image_embeddings, text_embeddings)

# Evaluate results
print(cosine_similarities)
