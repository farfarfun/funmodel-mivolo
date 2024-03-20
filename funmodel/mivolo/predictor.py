import numpy as np
from fundrive.download import simple_download
from fundrive.drives.oss import public_oss_url
from funmodel.core.predict.image import ImagePredictModel
from funmodel.mivolo.model.mi_volo import MiVOLO
from funmodel.mivolo.model.yolo_detector import Detector
from funmodel.mivolo.structures import PersonAndFaceResult


class MivoloPredictor(ImagePredictModel):
    def __init__(
        self,
        with_persons=False,
        disable_faces=False,
        device="cpu",
        verbose: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(model_name="mivolo", *args, **kwargs)
        self.detector_model = None
        self.age_gender_model = None
        self.load(with_persons, disable_faces, device, verbose)

    def load(
        self,
        with_persons=False,
        disable_faces=False,
        device="cpu",
        verbose: bool = False,
        *args,
        **kwargs,
    ):
        detector_weights = f"{self.cache_path}/mivolo_imbd.pth.tar"
        checkpoint = f"{self.cache_path}/yolov8x_person_face.pt"
        simple_download(
            url=public_oss_url(path="models/mivolo/mivolo_imbd.pth.tar"),
            filepath=detector_weights,
        )
        simple_download(
            url=public_oss_url(path="models/mivolo/yolov8x_person_face.pt"),
            filepath=checkpoint,
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

    def predict(self, image: np.ndarray, draw=False, *args, **kwargs):
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
