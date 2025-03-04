import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import pandas as pd
import numpy as np
from PIL import Image
import os

class ImageCaptioningDataset(Dataset):
    def __init__(self, caption_file, image_dir, feature_extractor, tokenizer, max_length=128):
        self.df = pd.read_csv(caption_file)
        self.image_dir = image_dir
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_dir, row['image'])
        caption = row['caption']

        image = Image.open(image_path).convert('RGB')
        
        # Process image
        pixel_values = self.feature_extractor(image, return_tensors='pt').pixel_values
        
        # Process text
        tokenized = self.tokenizer(caption, padding='max_length', max_length=self.max_length, truncation=True, return_tensors='pt')
        
        return {
            'pixel_values': pixel_values.squeeze(),
            'labels': tokenized.input_ids.squeeze()
        }

def load_model_and_processors():
    model_name = 'nlpconnect/vit-gpt2-image-captioning'
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    feature_extractor = ViTImageProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    return model, feature_extractor, tokenizer

def generate_caption(image_path, model, feature_extractor, tokenizer):
    image = Image.open(image_path).convert('RGB')
    pixel_values = feature_extractor(image, return_tensors='pt').pixel_values

    generated_ids = model.generate(
        pixel_values,
        max_length=30,
        num_beams=4,
        early_stopping=True
    )

    generated_caption = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_caption

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    caption_file = os.path.join(base_dir, 'captions.txt')
    image_dir = os.path.join(base_dir, 'Images')

    print("Loading model and processors...")
    model, feature_extractor, tokenizer = load_model_and_processors()

    # Test the model on a few images
    print("\nGenerating captions for sample images:")
    sample_images = os.listdir(image_dir)[:5]
    
    for image_name in sample_images:
        image_path = os.path.join(image_dir, image_name)
        caption = generate_caption(image_path, model, feature_extractor, tokenizer)
        print(f"\nImage: {image_name}")
        print(f"Generated caption: {caption}")

if __name__ == '__main__':
    main()