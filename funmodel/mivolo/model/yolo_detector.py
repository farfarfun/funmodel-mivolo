import os

import numpy as np
import torch
from funmodel.mivolo.structures import PersonAndFaceResult
from PIL import Image
from ultralytics.yolo.engine.model import YOLO
from ultralytics.yolo.engine.results import Results

# because of ultralytics bug it is important to unset CUBLAS_WORKSPACE_CONFIG after the module importing
os.unsetenv("CUBLAS_WORKSPACE_CONFIG")


class Detector:
    """基于 YOLOv8 的人脸/人体检测器，封装模型加载、单帧检测与多帧跟踪。"""

    def __init__(
        self,
        weights: str,
        device: str = "cuda",
        half: bool = True,
        verbose: bool = False,
        conf_thresh: float = 0.4,
        iou_thresh: float = 0.7,
    ) -> None:
        """加载 YOLOv8 权重并初始化检测参数。

        Args:
            weights: YOLOv8 权重文件路径。
            device: 推理设备，如 ``"cuda"`` 或 ``"cpu"``。
            half: 是否使用半精度推理（``device`` 为 CPU 时自动关闭）。
            verbose: 是否输出 YOLO 推理过程日志。
            conf_thresh: 检测置信度阈值。
            iou_thresh: NMS 的 IoU 阈值。
        """
        self.yolo = YOLO(weights)
        self.yolo.fuse()

        self.device = torch.device(device)
        self.half = half and self.device.type != "cpu"

        if self.half:
            self.yolo.model = self.yolo.model.half()

        self.detector_names: dict[int, str] = self.yolo.model.names

        # init yolo.predictor
        self.detector_kwargs = {"conf": conf_thresh, "iou": iou_thresh, "half": self.half, "verbose": verbose}
        # self.yolo.predict(**self.detector_kwargs)

    def predict(self, image: np.ndarray | str | Image.Image) -> PersonAndFaceResult:
        """对单张图片做人脸/人体检测，返回封装后的检测结果。

        Args:
            image: 输入图片，可以是 BGR 数组、图片路径或 PIL 图片。

        Returns:
            封装了检测框、类别等信息的 ``PersonAndFaceResult``。
        """
        results: Results = self.yolo.predict(image, **self.detector_kwargs)[0]
        return PersonAndFaceResult(results)

    def track(self, image: np.ndarray | str | Image.Image) -> PersonAndFaceResult:
        """对视频帧做人脸/人体检测并保持跨帧的目标 ID（用于视频跟踪场景）。

        Args:
            image: 输入图片，可以是 BGR 数组、图片路径或 PIL 图片。

        Returns:
            封装了检测框、跟踪 ID 等信息的 ``PersonAndFaceResult``。
        """
        results: Results = self.yolo.track(image, persist=True, **self.detector_kwargs)[0]
        return PersonAndFaceResult(results)
