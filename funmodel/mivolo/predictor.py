from typing import Any

import numpy as np
from funget import simple_download
from fundrive.drives.oss import public_oss_url
from funmodel.core.predict.image import ImagePredictModel
from funmodel.mivolo.model.mi_volo import MiVOLO
from funmodel.mivolo.model.yolo_detector import Detector
from funmodel.mivolo.structures import PersonAndFaceResult


class MivoloPredictor(ImagePredictModel):
    """基于 MiVOLO 的人脸/人体年龄与性别预测器。

    首次 ``load`` 会自动从组织 OSS 下载检测器权重（YOLOv8 人脸/人体检测）
    和 MiVOLO 年龄性别分类权重到 ``self.cache_path``。
    """

    def __init__(
        self,
        with_persons: bool = False,
        disable_faces: bool = False,
        device: str = "cpu",
        verbose: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """初始化预测器并立即加载模型权重。

        Args:
            with_persons: 是否同时使用人体框辅助年龄/性别判断。
            disable_faces: 是否禁用人脸检测，仅使用人体框。
            device: 推理设备，如 "cpu" 或 "cuda:0"。
            verbose: 是否输出详细日志。
        """
        super().__init__(model_name="mivolo", *args, **kwargs)
        self.detector_model: Detector | None = None
        self.age_gender_model: MiVOLO | None = None
        self.load(with_persons, disable_faces, device, verbose)

    def load(
        self,
        with_persons: bool = False,
        disable_faces: bool = False,
        device: str = "cpu",
        verbose: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """下载并加载检测器与年龄性别分类模型权重。

        Args:
            with_persons: 是否同时使用人体框辅助年龄/性别判断。
            disable_faces: 是否禁用人脸检测，仅使用人体框。
            device: 推理设备，如 "cpu" 或 "cuda:0"。
            verbose: 是否输出详细日志。
        """
        checkpoint = f"{self.cache_path}/mivolo_imbd.pth.tar"
        detector_weights = f"{self.cache_path}/yolov8x_person_face.pt"
        simple_download(
            url=public_oss_url(path="models/mivolo/mivolo_imbd.pth.tar"),
            filepath=checkpoint,
        )
        simple_download(
            url=public_oss_url(path="models/mivolo/yolov8x_person_face.pt"),
            filepath=detector_weights,
        )

        self.detector_model = Detector(detector_weights, device, verbose=verbose)
        self.age_gender_model = MiVOLO(
            checkpoint,
            device,
            half=True,
            use_persons=with_persons,
            disable_faces=disable_faces,
            verbose=verbose,
        )

    def predict(
        self, image: np.ndarray, draw: bool = False, *args: Any, **kwargs: Any
    ) -> tuple[list[dict[str, Any]], np.ndarray | None]:
        """对单张图片做人脸检测 + 年龄/性别预测。

        Args:
            image: BGR 格式的图片数组（如 ``cv2.imread`` 的返回值）。
            draw: 是否额外返回画好检测框和标签的图片。

        Returns:
            一个二元组：
            - 每个检测到的人脸对应一个字典，包含 age/gender/gender_score/body/cls；
            - 当 ``draw=True`` 时为标注后的图片数组，否则为 ``None``。
        """
        detected_objects: PersonAndFaceResult = self.detector_model.predict(image)
        self.age_gender_model.predict(image, detected_objects)

        out_im = None
        if draw:
            out_im = detected_objects.plot()
        result = []
        for i in range(detected_objects.n_objects):
            result.append(
                {
                    "age": detected_objects.ages[i],
                    "gender": detected_objects.genders[i],
                    "gender_score": detected_objects.gender_scores[i],
                    "body": detected_objects.yolo_results[i]
                    .boxes.xywh.cpu()
                    .numpy()[0]
                    .tolist(),
                    "cls": detected_objects.yolo_results[i]
                    .boxes.cls.cpu()
                    .numpy()[0]
                    .tolist(),
                }
            )
        result = [res for res in result if res["cls"] == 1]
        return result, out_im
