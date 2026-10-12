import os
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
from farlog import getLogger

_logger = getLogger("funmodel.mivolo.data_reader")

IMAGES_EXT: tuple = (".jpeg", ".jpg", ".png", ".webp", ".bmp", ".gif")
VIDEO_EXT: tuple = (".mp4", ".avi", ".mov", ".mkv", ".webm")


@dataclass
class PictureInfo:
    """保存单张图片的人脸、人体框及年龄性别标注。

    Args:
        image_path: 图片文件路径。
        age: 年龄、年龄区间或 ``"-1"``。
        gender: 性别标注或 ``"-1"``。
        bbox: 人脸框，坐标格式为 ``xyxy``。
        person_bbox: 人体框，坐标格式为 ``xyxy``。
    """

    image_path: str
    age: str | None  # age or age range(start;end format) or "-1"
    gender: str | None  # "M" of "F" or "-1"
    bbox: list[int] = field(default_factory=lambda: [-1, -1, -1, -1])  # face bbox: xyxy
    person_bbox: list[int] = field(default_factory=lambda: [-1, -1, -1, -1])  # person bbox: xyxy

    @property
    def has_person_bbox(self) -> bool:
        return any(coord != -1 for coord in self.person_bbox)

    @property
    def has_face_bbox(self) -> bool:
        return any(coord != -1 for coord in self.bbox)

    def has_gt(self, only_age: bool = False) -> bool:
        if only_age:
            return self.age != "-1"
        else:
            return not (self.age == "-1" and self.gender == "-1")

    def clear_person_bbox(self) -> None:
        self.person_bbox = [-1, -1, -1, -1]

    def clear_face_bbox(self) -> None:
        self.bbox = [-1, -1, -1, -1]


class AnnotType(Enum):
    ORIGINAL = "original"
    PERSONS = "persons"
    NONE = "none"

    @classmethod
    def _missing_(cls, value):
        _logger.warning(f"Unknown annotation type {value}.")
        return AnnotType.NONE


def get_all_files(path: str, extensions: tuple[str, ...] = IMAGES_EXT) -> list[str]:
    """递归获取目录中具有指定扩展名的文件。

    Args:
        path: 待遍历的目录路径。
        extensions: 允许的文件扩展名。

    Returns:
        所有匹配文件的路径列表。
    """
    files_all: list[str] = []
    for root, subFolders, files in os.walk(path):
        for name in files:
            # linux tricks with .directory that still is file
            if "directory" not in name and sum([ext.lower() in name.lower() for ext in extensions]) > 0:
                files_all.append(os.path.join(root, name))
    return files_all


class InputType(Enum):
    Image = 0
    Video = 1
    VideoStream = 2


def get_input_type(input_path: str) -> InputType:
    if os.path.isdir(input_path):
        _logger.info("Input is a folder, only images will be processed")
        return InputType.Image
    elif os.path.isfile(input_path):
        if input_path.endswith(VIDEO_EXT):
            return InputType.Video
        if input_path.endswith(IMAGES_EXT):
            return InputType.Image
        else:
            raise ValueError(
                f"Unknown or unsupported input file format {input_path}, \
                             supported video formats: {VIDEO_EXT}, \
                             supported image formats: {IMAGES_EXT}"
            )
    elif input_path.startswith("http") and not input_path.endswith(IMAGES_EXT):
        return InputType.VideoStream
    else:
        raise ValueError(f"Unknown input {input_path}")


def read_csv_annotation_file(
    annotation_file: str, images_dir: str, ignore_without_gt: bool = False
) -> tuple[dict[str, list[PictureInfo]], AnnotType]:
    """读取 MiVOLO 格式 CSV 标注并关联存在的图片。

    Args:
        annotation_file: CSV 标注文件路径。
        images_dir: 图片根目录。
        ignore_without_gt: 是否忽略缺少年龄或性别真值的记录。

    Returns:
        按图片路径分组的标注，以及检测到的标注类型。
    """
    bboxes_per_image: dict[str, list[PictureInfo]] = defaultdict(list)

    df = pd.read_csv(annotation_file, sep=",")

    annot_type = AnnotType("persons") if "person_x0" in df.columns else AnnotType("original")
    _logger.info(f"Reading {annotation_file} (type: {annot_type})...")

    missing_images = 0
    for index, row in df.iterrows():
        img_path = os.path.join(images_dir, row["img_name"])
        if not os.path.exists(img_path):
            missing_images += 1
            continue

        face_x1, face_y1, face_x2, face_y2 = row["face_x0"], row["face_y0"], row["face_x1"], row["face_y1"]
        age, gender = str(row["age"]), str(row["gender"])

        if ignore_without_gt and (age == "-1" or gender == "-1"):
            continue

        if annot_type == AnnotType.PERSONS:
            p_x1, p_y1, p_x2, p_y2 = row["person_x0"], row["person_y0"], row["person_x1"], row["person_y1"]
            person_bbox = list(map(int, [p_x1, p_y1, p_x2, p_y2]))
        else:
            person_bbox = [-1, -1, -1, -1]

        bbox = list(map(int, [face_x1, face_y1, face_x2, face_y2]))
        pic_info = PictureInfo(img_path, age, gender, bbox, person_bbox)
        bboxes_per_image[img_path].append(pic_info)

    if missing_images > 0:
        _logger.warning(f"Missing images: {missing_images}/{len(df)}")
    return bboxes_per_image, annot_type
