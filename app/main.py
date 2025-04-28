import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["JOBLIB_NUM_THREADS"] = "1"

from itertools import chain
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import torch
from model.clip_loader import load_clip_model
from model.aesthetic_regressor import loader_aesthetic_regressor
from utils.image_loader import load_images
from core.cache import get_cached_embedding, get_cached_embeddings_parallel
from schemas.album_schema import ImageRequest, CategoryScoreRequest, ImageCategoryGroup
from service.embedding import embed_images
from service.category import categorize_images
from service.duplicate import find_duplicate_groups
from service.highlight import estimate_highlight_score

app = FastAPI()
torch.set_num_threads(1)

clip_model = None
clip_preprocess = None
aesthetic_regressor = None

@app.on_event("startup")
def load():
    global clip_model, clip_preprocess, aesthetic_regressor
    clip_model, clip_preprocess = load_clip_model()
    aesthetic_regressor = loader_aesthetic_regressor()

@app.get("/")
def ping():
    return {"ping": 'success'}

@app.post("/api/albums/embedding", status_code=201)
def embed(req: ImageRequest):
    global clip_model, clip_preprocess
    filenames = req.images
    images = load_images(filenames)
    embed_images(clip_model, clip_preprocess, images, filenames, batch_size=16, device='cpu')
    return {"message": "success", "data": None}
    
@app.post("/api/albums/categories", status_code=201)
def categorize(req: ImageRequest):
    data = torch.load("/Users/images/category_features.pt", weights_only=True)
    categories = data["categories"]
    text_features = data["text_features"]
    image_names = req.images
    image_features, missing_keys = get_cached_embeddings_parallel(image_names)
    if len(missing_keys) > 0:
        return JSONResponse(status_code=428, content={"message": "embedding_required", "data": missing_keys})
    image_features = torch.stack([get_cached_embedding(image_name) for image_name in image_names])
    image_features /= image_features.norm(dim=-1, keepdim=True)    
    categorized = categorize_images(image_features.cpu(), image_names, text_features.cpu(), categories)
    response = [ImageCategoryGroup(category=category, images=images) for category, images in categorized.items()]
    
    return {"message": "success", "data": response}
    
@app.post("/api/albums/duplicates")
def duplicate(req: ImageRequest):
    image_names = req.images
    image_features, missing_keys = get_cached_embeddings_parallel(image_names)
    if len(missing_keys) > 0:
        return JSONResponse(status_code=428, content={"message": "embedding_required", "data": missing_keys})
    image_features = torch.stack([get_cached_embedding(image_name) for image_name in image_names])
    image_features /= image_features.norm(dim=-1, keepdim=True)    
    duplicate_image_groups = find_duplicate_groups(image_features, image_names)
    return {"message": "success", "data": duplicate_image_groups}
    
@app.post("/api/albums/scores")
def highlight_scoring(req: CategoryScoreRequest):    
    categories = req.categories
    all_images = list(chain.from_iterable(category.images for category in categories))
    image_features, missing_keys = get_cached_embeddings_parallel(all_images)
    if len(missing_keys) > 0:
        return JSONResponse(status_code=428, content={"message": "embedding_required", "data": missing_keys})
    embedding_map = {image: feature for image, feature in zip(all_images, image_features)}
        
    data = []
    for category in categories:
        image_features = torch.stack([embedding_map[image] for image in category.images])
        image_features /= image_features.norm(dim=-1, keepdim=True)    
        scores = estimate_highlight_score(image_features, category.images, aesthetic_regressor)
        data.append({"category": category.category, "images": scores})
        
    return JSONResponse(status_code=201, content={"message": "success", "data": data})