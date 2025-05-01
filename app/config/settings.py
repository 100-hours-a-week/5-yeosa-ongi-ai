import os
from enum import Enum
from dotenv import load_dotenv

load_dotenv()

class ImageMode(str, Enum):
    LOCAL = "local"
    GCS = "gcs"

mode_str = os.getenv("IMAGE_MODE", "gcs")

try:
    IMAGE_MODE = ImageMode(mode_str)
except ValueError:
    raise ValueError(f"잘못된 IMAGE_MODE: {mode_str}. 선택 가능한 IMAGE_MODE: {[m.value for m in ImageMode]}")
