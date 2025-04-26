import os
from PIL import Image

def load_images(filenames: list[str]) -> list[Image.Image]:
    images_path = '/Users/images' 
    images = [Image.open(os.path.join(images_path, filename)).convert('RGB') for filename in filenames]
    return images