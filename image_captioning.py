import pandas as pd
import os
from PIL import Image

def load_captions(caption_file):
    """Load and process image captions from the file"""
    df = pd.read_csv(caption_file)
    # Group captions by image
    caption_dict = {}
    for _, row in df.iterrows():
        if row['image'] not in caption_dict:
            caption_dict[row['image']] = []
        caption_dict[row['image']].append(row['caption'])
    return caption_dict

def process_image(image_path):
    """Load and process an image"""
    try:
        img = Image.open(image_path)
        return img
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None

def main():
    # Set paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    caption_file = os.path.join(base_dir, 'captions.txt')
    image_dir = os.path.join(base_dir, 'Images')
    
    # Load captions
    print("Loading captions...")
    caption_dict = load_captions(caption_file)
    
    # Process a few images as an example
    print("\nProcessing sample images:")
    for image_name in list(caption_dict.keys())[:5]:
        image_path = os.path.join(image_dir, image_name)
        if os.path.exists(image_path):
            img = process_image(image_path)
            if img:
                print(f"\nImage: {image_name}")
                print("Captions:")
                for caption in caption_dict[image_name]:
                    print(f"- {caption}")

if __name__ == '__main__':
    main()