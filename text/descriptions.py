from transformers import AutoProcessor, AutoModelForPreTraining
import torch
import os
from PIL import Image
import csv

llava_model = "llava-hf/llava-interleave-qwen-0.5b-hf"
processor = AutoProcessor.from_pretrained(llava_model)
device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
model = AutoModelForPreTraining.from_pretrained(llava_model).to(device)

def extract_images_of_models(folder_path, output_dir):
    # Define the column headers
    headers = ['Class', 'Subclass', 'Description']
    # Open a file in write mode (use 'a' for appending to existing file)
    with open(output_dir, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for classes in os.listdir(folder_path):
            class_dir = os.path.join(folder_path, classes)
            for models in os.listdir(class_dir):
                model_dir = os.path.join(class_dir, models)
                for images in os.listdir(model_dir):
                    image_path = os.path.join(model_dir, images)
                    torch.cuda.empty_cache()  # Clear the cache
                    # Prepare image and text prompt
                    image = Image.open(image_path).convert('RGB')
                    prompt = f"user<image>Describe the features of the focussed object in the image?assistant:"
                    inputs = processor(prompt, image, return_tensors="pt").to(device)
                    torch.cuda.empty_cache()  # Clear the cache before generating output
                    output = model.generate(**inputs, max_new_tokens=100)
                    row = {'Class': f'{classes}', 'Subclasses': f'{models}', 'Description': f'{str(processor.decode(output[0], skip_special_tokens=True))[73:]}'}
                    writer.writerow(row)
                torch.cuda.empty_cache()

images_pwd = 'C:/Project/GPTImages'
output_dir = './descriptions.csv'
images = extract_images_of_models(images_pwd, output_dir)