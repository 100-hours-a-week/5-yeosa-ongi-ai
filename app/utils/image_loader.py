import os
from PIL import Image
from abc import ABC, abstractmethod

class BaseImageLoader(ABC):
    @abstractmethod
    def load_images(self) -> list[Image.Image]:
        pass

class LocalImageLoader(BaseImageLoader):
    def __init__(self, image_dir: str = "Users/images"):
        self.image_dir = image_dir 

    def load_images(self, filenames: list[str]) -> list[Image.Image]:
        return [
            Image.open(os.path.join(self.image_dir, filename)).convert("RGB")
            for filename in filenames
        ]