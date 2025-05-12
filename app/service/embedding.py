import torch

from app.core.cache import set_cached_embedding
from app.utils.logging_decorator import log_exception, log_flow


@log_flow
def embed_images(
    model, preprocess, images, filenames, batch_size=16, device="cpu"
):
    preprocessed_img = [preprocess(image) for image in images]

    for i in range(0, len(images), batch_size):
        batch_images = preprocessed_img[i : i + batch_size]
        batch_filenames = filenames[i : i + batch_size]
        image_input = torch.stack(batch_images).to(device)
        with torch.no_grad():
            batch_features = model.encode_image(image_input).cpu()
        for filename, feature in zip(batch_filenames, batch_features):
            set_cached_embedding(filename, feature)

    return None