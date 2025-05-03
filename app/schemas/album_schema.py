from pydantic import BaseModel


# /embedding, /duplicate, /quality, /categories, /people 등에서 공통 사용
class ImageRequest(BaseModel):
    """
    이미지 파일명 목록을 전달하기 위한 요청 모델.

    Attributes:
        images (list[str]): 처리할 이미지 파일명 목록입니다.

    """

    images: list[str]


# /scores 요청용
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
