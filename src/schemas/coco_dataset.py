from pydantic import BaseModel, Field, HttpUrl
from datetime import datetime


class COCOInfo(BaseModel):
    description: str | None = None
    url: HttpUrl | None = None
    version: str | None = None
    year: int | None = None
    contributor: str | None = None
    date_created: datetime | None = None


class COCOLicenses(BaseModel):
    url: HttpUrl | None = None
    id: int
    name: str | None = None


class COCOImage(BaseModel):
    id: int
    license: int | None = None
    width: int
    height: int
    file_name: str
    date_captured: datetime


class COCOAnnotation(BaseModel):
    id: int
    category_id: int
    iscrowd: int = Field(0, ge=0, le=1)
    segmentation: list[list[float]]
    image_id: int
    area: float
    bbox: list[float] = Field(min_length=4, max_length=4)


class COCOCategory(BaseModel):
    supercategory: str | None = None
    id: int
    name: str


class COCODataset(BaseModel):
    info: COCOInfo | None = None
    licenses: list[COCOLicenses] | None = None
    images: list[COCOImage]
    annotations: list[COCOAnnotation]
    categories: list[COCOCategory]
