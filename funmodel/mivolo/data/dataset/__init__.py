import torch
from funmodel.mivolo.model.mi_volo import MiVOLO

from .age_gender_dataset import AgeGenderDataset
from .age_gender_loader import create_loader
from .classification_dataset import AdienceDataset, FairFaceDataset

DATASET_CLASS_MAP = {
    "utk": AgeGenderDataset,
    "lagenda": AgeGenderDataset,
    "imdb": AgeGenderDataset,
    "agedb": AgeGenderDataset,
    "cacd": AgeGenderDataset,
    "adience": AdienceDataset,
    "fairface": FairFaceDataset,
}


def build(
    name: str,
    images_path: str,
    annotations_path: str,
    split: str,
    mivolo_model: MiVOLO,
    workers: int,
    batch_size: int,
) -> tuple[torch.utils.data.Dataset, torch.utils.data.DataLoader]:
    """按数据集名称构建训练/评估用的 ``Dataset`` 与 ``DataLoader``。

    Args:
        name: 数据集名称，需是 ``DATASET_CLASS_MAP`` 中已注册的 key。
        images_path: 图片根目录。
        annotations_path: 标注文件路径。
        split: 数据划分，如 ``"train"``/``"val"``/``"test"``。
        mivolo_model: 已加载的 ``MiVOLO`` 模型，用于读取输入尺寸、年龄范围等元信息。
        workers: ``DataLoader`` 的 worker 进程数。
        batch_size: 批大小。

    Returns:
        ``(dataset, dataset_loader)`` 二元组。
    """

    dataset_class = DATASET_CLASS_MAP[name]

    dataset: torch.utils.data.Dataset = dataset_class(
        images_path=images_path,
        annotations_path=annotations_path,
        name=name,
        split=split,
        target_size=mivolo_model.input_size,
        max_age=mivolo_model.meta.max_age,
        min_age=mivolo_model.meta.min_age,
        model_with_persons=mivolo_model.meta.with_persons_model,
        use_persons=mivolo_model.meta.use_persons,
        disable_faces=mivolo_model.meta.disable_faces,
        only_age=mivolo_model.meta.only_age,
    )

    data_config = mivolo_model.data_config

    in_chans = 3 if not mivolo_model.meta.with_persons_model else 6
    input_size = (in_chans, mivolo_model.input_size, mivolo_model.input_size)

    dataset_loader: torch.utils.data.DataLoader = create_loader(
        dataset,
        input_size=input_size,
        batch_size=batch_size,
        mean=data_config["mean"],
        std=data_config["std"],
        num_workers=workers,
        crop_pct=data_config["crop_pct"],
        crop_mode=data_config["crop_mode"],
        pin_memory=False,
        device=mivolo_model.device,
        target_type=dataset.target_dtype,
    )

    return dataset, dataset_loader
