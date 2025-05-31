from pydantic import BaseModel


# /embedding, /duplicate, /quality, /people 등에서 공통 사용
class ImageRequest(BaseModel):
    """
    이미지 파일명 목록을 전달하기 위한 요청 모델.

    Attributes:
        images (list[str]): 처리할 이미지 파일명 목록입니다.

    """

    images: list[str]
    
    
# /categories 요청용
class ImageConceptRequest(BaseModel):
    """
    카테고리를 분류할 이미지들을 요청하기 위한 모델.

    Attributes:
        concepts (list[str]): 사용자가 요청한 앨범 컨셉 목록입니다. 이에 따라 세부 카테고리가 달라집니다.
        images (list[str]): 해당 카테고리에 속하는 이미지 파일명 목록입니다.

    """
    concepts: list[str]
    images: list[str]


# /score 요청용
class ImageCategoryGroup(BaseModel):
    """
    카테고리별 이미지 묶음을 나타내는 모델.

    Attributes:
        category (str): 이미지 그룹의 카테고리 이름입니다.
        images (list[str]): 해당 카테고리에 속하는 이미지 파일명 목록입니다.

    """

    category: str
    images: list[str]


class CategoryScoreRequest(BaseModel):
    categories: list[ImageCategoryGroup]
