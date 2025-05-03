import torch

from app.core.cache import set_cached_embedding
from app.utils.logging_decorator import log_exception


@log_exception
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


# from core.cache import get_cached_embedding, set_cached_embedding
# import torch

# def embed_images(model, preprocess, images, batch_size=16, device='cpu'):
#     embeddings = []
#     filenames = []

#     # 캐시가 없는 이미지만 추림
#     uncached_items = []
#     for filename, image in images:
#         cached = get_cached_embedding(filename)
#         if cached is not None:
#             embeddings.append(cached.unsqueeze(0))  # [1, D]
#             filenames.append(filename)
#         else:
#             uncached_items.append((filename, image))

#     # 새로 임베딩해야 할 이미지만 처리
#     if uncached_items:
#         preprocessed_batch = [preprocess(image) for _, image in uncached_items]

#         for i in range(0, len(preprocessed_batch), batch_size):
#             batch_filenames = [fn for fn, _ in uncached_items[i:i+batch_size]]
#             batch = preprocessed_batch[i:i+batch_size]
#             image_input = torch.stack(batch).to(device)

#             with torch.no_grad():
#                 batch_features = model.encode_image(image_input).cpu()

#             # 캐시에 저장 & 결과에 추가
#             for filename, feature in zip(batch_filenames, batch_features):
#                 set_cached_embedding(filename, feature)
#                 embeddings.append(feature.unsqueeze(0))
#                 filenames.append(filename)

#     # 최종 결과 결합
#     if embeddings:
#         embeddings_tensor = torch.cat(embeddings, dim=0)  # [N, D]
#     else:
#         embeddings_tensor = torch.empty(0, model.visual.output_dim)

#     return embeddings_tensor, filenames
