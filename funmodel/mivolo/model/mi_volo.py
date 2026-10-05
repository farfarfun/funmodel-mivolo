import numpy as np
import torch
from farlog import getLogger
from funmodel.mivolo.data.misc import prepare_classification_images
from funmodel.mivolo.model.create_timm_model import create_model
from funmodel.mivolo.structures import PersonAndFaceCrops, PersonAndFaceResult
from timm.data import resolve_data_config

_logger = getLogger("MiVOLO")
has_compile = hasattr(torch, "compile")


class Meta:
    def __init__(self):
        self.min_age = None
        self.max_age = None
        self.avg_age = None
        self.num_classes = None

        self.in_chans = 3
        self.with_persons_model = False
        self.disable_faces = False
        self.use_persons = True
        self.only_age = False

        self.num_classes_gender = 2
        self.input_size = 224

    def load_from_ckpt(self, ckpt_path: str, disable_faces: bool = False, use_persons: bool = True) -> "Meta":

        state = torch.load(ckpt_path, map_location="cpu")

        self.min_age = state["min_age"]
        self.max_age = state["max_age"]
        self.avg_age = state["avg_age"]
        self.only_age = state["no_gender"]

        only_age = state["no_gender"]

        self.disable_faces = disable_faces
        if "with_persons_model" in state:
            self.with_persons_model = state["with_persons_model"]
        else:
            self.with_persons_model = True if "patch_embed.conv1.0.weight" in state["state_dict"] else False

        self.num_classes = 1 if only_age else 3
        self.in_chans = 3 if not self.with_persons_model else 6
        self.use_persons = use_persons and self.with_persons_model

        if not self.with_persons_model and self.disable_faces:
            raise ValueError("You can not use disable-faces for faces-only model")
        if self.with_persons_model and self.disable_faces and not self.use_persons:
            raise ValueError(
                "You can not disable faces and persons together. "
                "Set --with-persons if you want to run with --disable-faces"
            )
        self.input_size = state["state_dict"]["pos_embed"].shape[1] * 16
        return self

    def __str__(self):
        attrs = vars(self)
        attrs.update({"use_person_crops": self.use_person_crops, "use_face_crops": self.use_face_crops})
        return ", ".join("%s: %s" % item for item in attrs.items())

    @property
    def use_person_crops(self) -> bool:
        return self.with_persons_model and self.use_persons

    @property
    def use_face_crops(self) -> bool:
        return not self.disable_faces or not self.with_persons_model


class MiVOLO:
    """MiVOLO 年龄/性别分类模型封装：加载权重、预处理裁剪图并写回预测结果。"""

    def __init__(
        self,
        ckpt_path: str,
        device: str = "cuda",
        half: bool = True,
        disable_faces: bool = False,
        use_persons: bool = True,
        verbose: bool = False,
        torchcompile: str | None = None,
    ) -> None:
        """从 checkpoint 加载 MiVOLO 模型并完成推理前的初始化。

        Args:
            ckpt_path: MiVOLO 权重文件路径（包含 ``min_age``/``max_age`` 等元信息）。
            device: 推理设备，如 ``"cuda"`` 或 ``"cpu"``。
            half: 是否使用半精度推理（``device`` 为 CPU 时自动关闭）。
            disable_faces: 是否禁用人脸输入，仅使用人体裁剪。
            use_persons: 是否同时使用人体裁剪辅助判断（需模型本身支持）。
            verbose: 是否输出模型元信息日志。
            torchcompile: 传给 ``torch.compile`` 的 backend 名称，为空则不编译。
        """
        self.verbose = verbose
        self.device = torch.device(device)
        self.half = half and self.device.type != "cpu"

        self.meta: Meta = Meta().load_from_ckpt(ckpt_path, disable_faces, use_persons)
        if self.verbose:
            _logger.info(f"Model meta:\n{str(self.meta)}")

        model_name = f"mivolo_d1_{self.meta.input_size}"
        self.model = create_model(
            model_name=model_name,
            num_classes=self.meta.num_classes,
            in_chans=self.meta.in_chans,
            pretrained=False,
            checkpoint_path=ckpt_path,
            filter_keys=["fds."],
        )
        self.param_count = sum([m.numel() for m in self.model.parameters()])
        _logger.info(f"Model {model_name} created, param count: {self.param_count}")

        self.data_config = resolve_data_config(
            model=self.model,
            verbose=verbose,
            use_test_size=True,
        )

        self.data_config["crop_pct"] = 1.0
        c, h, w = self.data_config["input_size"]
        assert h == w, "Incorrect data_config"
        self.input_size = w

        self.model = self.model.to(self.device)

        if torchcompile:
            assert has_compile, "A version of torch w/ torch.compile() is required for --compile, possibly a nightly."
            torch._dynamo.reset()
            self.model = torch.compile(self.model, backend=torchcompile)

        self.model.eval()
        if self.half:
            self.model = self.model.half()

    def warmup(self, batch_size: int, steps: int = 10) -> None:
        """用随机输入跑若干次前向推理，用于预热 CUDA kernel、稳定首帧延迟。

        Args:
            batch_size: 预热用的 batch 大小。
            steps: 预热迭代次数。
        """
        if self.meta.with_persons_model:
            input_size = (6, self.input_size, self.input_size)
        else:
            input_size = self.data_config["input_size"]

        input = torch.randn((batch_size,) + tuple(input_size)).to(self.device)

        for _ in range(steps):
            out = self.inference(input)  # noqa: F841

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def inference(self, model_input: torch.tensor) -> torch.tensor:
        """对预处理后的输入张量做一次前向推理，返回模型原始输出。"""

        with torch.no_grad():
            if self.half:
                model_input = model_input.half()
            output = self.model(model_input)
        return output

    def predict(self, image: np.ndarray, detected_bboxes: PersonAndFaceResult) -> None:
        """对检测到的人脸/人体裁剪图做年龄与性别预测，结果写回 ``detected_bboxes``。

        Args:
            image: 原始 BGR 图片，用于裁剪人脸/人体区域。
            detected_bboxes: ``Detector`` 输出的检测结果，预测值原地写入其
                ``ages``/``genders``/``gender_scores``。
        """
        if (
            (detected_bboxes.n_objects == 0)
            or (not self.meta.use_persons and detected_bboxes.n_faces == 0)
            or (self.meta.disable_faces and detected_bboxes.n_persons == 0)
        ):
            # nothing to process
            return

        faces_input, person_input, faces_inds, bodies_inds = self.prepare_crops(image, detected_bboxes)

        if faces_input is None and person_input is None:
            # nothing to process
            return

        if self.meta.with_persons_model:
            model_input = torch.cat((faces_input, person_input), dim=1)
        else:
            model_input = faces_input
        output = self.inference(model_input)

        # write gender and age results into detected_bboxes
        self.fill_in_results(output, detected_bboxes, faces_inds, bodies_inds)

    def fill_in_results(
        self,
        output: torch.Tensor,
        detected_bboxes: PersonAndFaceResult,
        faces_inds: list[int | None],
        bodies_inds: list[int | None],
    ) -> None:
        """将模型输出反归一化为年龄/性别，写入 ``detected_bboxes`` 对应下标。

        年龄反归一化公式为 ``age = raw_age * (max_age - min_age) + avg_age``，
        三个常量均来自训练时写入 checkpoint 的 ``self.meta``。

        Args:
            output: 模型前向输出（age 或 age+gender logits）。
            detected_bboxes: 待写入预测结果的检测结果对象。
            faces_inds: 每条输出对应的人脸下标（可能为 ``None``）。
            bodies_inds: 每条输出对应的人体下标（可能为 ``None``）。
        """
        if self.meta.only_age:
            age_output = output
            gender_probs, gender_indx = None, None
        else:
            age_output = output[:, 2]
            gender_output = output[:, :2].softmax(-1)
            gender_probs, gender_indx = gender_output.topk(1)

        assert output.shape[0] == len(faces_inds) == len(bodies_inds)

        # per face
        for index in range(output.shape[0]):
            face_ind = faces_inds[index]
            body_ind = bodies_inds[index]

            # get_age
            age = age_output[index].item()
            age = age * (self.meta.max_age - self.meta.min_age) + self.meta.avg_age
            age = round(age, 2)

            detected_bboxes.set_age(face_ind, age)
            detected_bboxes.set_age(body_ind, age)

            _logger.info(f"\tage: {age}")

            if gender_probs is not None:
                gender = "male" if gender_indx[index].item() == 0 else "female"
                gender_score = gender_probs[index].item()

                _logger.info(f"\tgender: {gender} [{int(gender_score * 100)}%]")

                detected_bboxes.set_gender(face_ind, gender, gender_score)
                detected_bboxes.set_gender(body_ind, gender, gender_score)

    def prepare_crops(
        self, image: np.ndarray, detected_bboxes: PersonAndFaceResult
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, list[int | None], list[int | None]]:
        """从原图裁剪出人脸/人体区域并做模型输入所需的归一化预处理。

        Args:
            image: 原始 BGR 图片。
            detected_bboxes: 已完成人脸-人体关联的检测结果。

        Returns:
            ``(人脸输入张量, 人体输入张量, 人脸下标列表, 人体下标列表)``；
            未启用对应分支时返回 ``None``。
        """

        if self.meta.use_person_crops and self.meta.use_face_crops:
            detected_bboxes.associate_faces_with_persons()

        crops: PersonAndFaceCrops = detected_bboxes.collect_crops(image)
        (bodies_inds, bodies_crops), (faces_inds, faces_crops) = crops.get_faces_with_bodies(
            self.meta.use_person_crops, self.meta.use_face_crops
        )

        if not self.meta.use_face_crops:
            assert all(f is None for f in faces_crops)

        faces_input = prepare_classification_images(
            faces_crops, self.input_size, self.data_config["mean"], self.data_config["std"], device=self.device
        )

        if not self.meta.use_person_crops:
            assert all(p is None for p in bodies_crops)

        person_input = prepare_classification_images(
            bodies_crops, self.input_size, self.data_config["mean"], self.data_config["std"], device=self.device
        )

        _logger.info(
            f"faces_input: {faces_input.shape if faces_input is not None else None}, "
            f"person_input: {person_input.shape if person_input is not None else None}"
        )

        return faces_input, person_input, faces_inds, bodies_inds


if __name__ == "__main__":
    model = MiVOLO("../pretrained/checkpoint-377.pth.tar", half=True, device="cuda:0")
