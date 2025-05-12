import time
import io
from PIL import Image
from app.utils.image_loader import GCSImageLoader

dumy_img = [
    "img001.jpg",
    "img002.jpg",
    "img003.jpg",
    "img004.jpg",
    "img005.jpg",
    "img006.jpg",
    "img007.jpg",
    "img008.jpg",
    "img009.jpg",
    "img010.jpg",
    "img011.jpg",
    "img012.jpg",
    "img013.jpg",
    "img014.jpg",
    "img015.jpg",
    "img016.jpg",
    "img017.jpg",
    "img018.jpg",
    "img019.jpg",
    "img020.jpg",
    "img021.jpg",
    "img022.jpg",
    "img023.jpg",
    "img024.jpg",
    "img025.jpg",
    "img026.jpg",
    "img027.jpg",
    "img028.jpg",
    "img029.jpg",
    "img030.jpg",
]


async def warm_up_gcs(
    image_loader: GCSImageLoader, filename: list[str] = dumy_img
) -> None:
    """
    GCS와 TurboJPEG 초기화 예열을 수행합니다.
    Args:
        image_loader (GCSImageLoader): GCS 이미지 로더
        filename (str): 예열용 이미지 파일 이름 (GCS 경로 기준)
    """
    try:
        print(f"[예열] GCS에서 {filename} 다운로드 시작")
        start = time.time()
        _ = await image_loader.load_images(dumy_img)

        print(f"[예열] 완료: {time.time() - start:.3f}s")

    except Exception as e:
        print(f"[예열 실패] {e}")
