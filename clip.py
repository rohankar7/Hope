from PIL import Image
import torch
import os
import pandas as pd
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
import pyvista as pv
import config
import torch.nn.functional as F

def render_generated_models():
    out_dir = './clip_images'
    os.makedirs(out_dir, exist_ok=True)
    generated_models_dir = f'./generated_models_{config.triplane_resolution}'
    generated_models_dir = f'./generated_models_256'
    for paths in os.listdir(generated_models_dir):
        mesh = pv.read(f'{generated_models_dir}/{paths}')

        mesh.set_active_scalars('RGBA')
        print(mesh.array_names) 
        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(mesh)
        plotter.show(screenshot=f'{out_dir}/{paths.split('.')[0]}.png')


def calculate_clip_score():
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = AutoModelForZeroShotImageClassification.from_pretrained("openai/clip-vit-base-patch32")
    df = pd.read_csv(config.captions)
    clip_images_dir = './clip_images'
    images = [Image.open(os.path.join(clip_images_dir, image)) for image in os.listdir(clip_images_dir)]
    texts = []
    for paths in os.listdir(clip_images_dir):
        path = paths.split('.')[0].split('_')
        c = f"'{path[0]}'"
        s = path[1]
        caption = df[(df['Class']==c) & (df['Subclass']==s)]['Caption'].iloc[0]
        texts.append(caption)
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    image_embeddings = outputs.image_embeds
    text_embeddings = outputs.text_embeds
    cosine_similarities = F.cosine_similarity(image_embeddings, text_embeddings)
    return cosine_similarities

def main():
    # render_generated_models()
    scores = calculate_clip_score()
    mean_score = torch.mean(scores).item()
    print('The CLIP score is:', mean_score)

if __name__ == '__main__':
    main()