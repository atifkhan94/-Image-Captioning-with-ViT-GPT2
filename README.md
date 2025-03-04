# Image Captioning with ViT-GPT2

This project implements an image captioning system using the Vision Transformer (ViT) and GPT-2 models. The model generates natural language descriptions for input images by combining the power of vision transformers for image understanding and GPT-2 for text generation.

## Features

- Uses ViT (Vision Transformer) as the image encoder
- Employs GPT-2 as the language model for caption generation
- Integrates with Hugging Face transformers library
- Provides example usage with sample images
- Includes a Jupyter notebook for interactive demonstration

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

- `image_captioning_model.py`: Main implementation of the image captioning model
- `image_captioning.py`: Utility script for processing images and captions
- `Image_Captioning_Demo.ipynb`: Interactive Jupyter notebook demonstration
- `Images/`: Directory containing sample images
- `captions.txt`: Caption annotations for the images
- `requirements.txt`: Python package dependencies

## Usage

### Using the Python Script

```python
from image_captioning_model import load_model_and_processors, generate_caption

# Load the model and processors
model, feature_extractor, tokenizer = load_model_and_processors()

# Generate caption for an image
image_path = 'path/to/your/image.jpg'
caption = generate_caption(image_path, model, feature_extractor, tokenizer)
print(f'Generated caption: {caption}')
```

### Using the Jupyter Notebook

1. Start Jupyter Notebook:
```bash
jupyter notebook
```
2. Open `Image_Captioning_Demo.ipynb`
3. Follow the interactive demonstration

## Model Details

The image captioning system uses:
- **Vision Transformer (ViT)**: For encoding images into meaningful representations
- **GPT-2**: For generating natural language captions
- Pre-trained model from Hugging Face: 'nlpconnect/vit-gpt2-image-captioning'

## Dependencies

- torch
- transformers
- Pillow
- pandas
- numpy

## License

This project is licensed under CC0-1.0 License.

## Acknowledgments

- Hugging Face for providing the pre-trained models
- The Vision Transformer and GPT-2 research teams