from functools import cached_property
from pathlib import Path

from pydantic import BaseModel, Field, HttpUrl, computed_field
from datetime import datetime
import json


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

    data_dir: Path | None = None

    @computed_field
    @cached_property
    def _image_map(self) -> dict[int, COCOImage]:
        return {img.id: img for img in self.images}

    @computed_field
    @cached_property
    def _annotations_map(self) -> dict[int, list[COCOAnnotation]]:
        mapping = {}
        for annot in self.annotations:
            mapping.setdefault(annot.image_id, []).append(annot)
        return mapping

    def get_data_by_image_id(self, image_id: int) -> "COCOImageResponse":
        image = self._image_map.get(image_id)
        if not image:
            raise KeyError(f"Изображение с id={image_id} не найдено в датасете.")

        annotations = self._annotations_map.get(image_id, [])

        return COCOImageResponse(image=image, annotations=annotations)

    def get_image_path(self, image_id) -> Path:
        image = self._image_map.get(image_id)
        if not image:
            raise FileNotFoundError("Изображение не найдено")

        path = self.data_dir / image.file_name

        if not path.is_file():
            raise FileNotFoundError("Изображение не найдено")

        return path

    @classmethod
    def load(cls, path: str | Path) -> "COCODataset":
        if not isinstance(path, Path):
            path = Path(path)

        if not path.is_dir():
            raise FileNotFoundError("Путь должен быть до директории")

        annotations_file = path / "annotations.json"
        if not annotations_file.is_file():
            raise FileNotFoundError("Файл аннотаций не найден")

        with open(annotations_file, mode="r") as file:
            data = json.load(file)

            dataset = cls(**data)
            dataset.data_dir = path
            return dataset


class COCOImageResponse(BaseModel):
    image: COCOImage
    annotations: list[COCOAnnotation]
