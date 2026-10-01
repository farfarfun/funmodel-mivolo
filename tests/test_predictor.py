from types import SimpleNamespace

import numpy as np
import pytest

import funmodel.mivolo.predictor as predictor_module
from funmodel.mivolo.predictor import MivoloPredictor


class FakeTensor:
    def __init__(self, value):
        self.value = np.asarray([value])

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class FakeDetectedObjects:
    def __init__(self, classes=()):
        self.n_objects = len(classes)
        self.ages = [30 + index for index in range(self.n_objects)]
        self.genders = ["female"] * self.n_objects
        self.gender_scores = [0.9] * self.n_objects
        self.yolo_results = [
            SimpleNamespace(
                boxes=SimpleNamespace(
                    xywh=FakeTensor([10, 20, 30, 40]),
                    cls=FakeTensor(class_id),
                )
            )
            for class_id in classes
        ]
        self.plotted = np.ones((2, 2, 3), dtype=np.uint8)

    def plot(self):
        return self.plotted


class FakeDetector:
    def __init__(self, detected_objects):
        self.detected_objects = detected_objects

    def predict(self, image):
        return self.detected_objects


class FakeAgeGenderModel:
    def __init__(self):
        self.calls = []

    def predict(self, image, detected_objects):
        self.calls.append((image, detected_objects))


def make_predictor(detected_objects):
    predictor = object.__new__(MivoloPredictor)
    predictor.detector_model = FakeDetector(detected_objects)
    predictor.age_gender_model = FakeAgeGenderModel()
    return predictor


def test_predict_returns_only_faces():
    detected_objects = FakeDetectedObjects(classes=(0, 1))
    predictor = make_predictor(detected_objects)
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    results, drawn_image = predictor.predict(image)

    assert drawn_image is None
    assert results == [
        {
            "age": 31,
            "gender": "female",
            "gender_score": 0.9,
            "body": [10, 20, 30, 40],
            "cls": 1,
        }
    ]
    assert predictor.age_gender_model.calls == [(image, detected_objects)]


def test_predict_handles_empty_detection_and_draw():
    detected_objects = FakeDetectedObjects()
    predictor = make_predictor(detected_objects)

    results, drawn_image = predictor.predict(np.zeros((4, 4, 3), dtype=np.uint8), draw=True)

    assert results == []
    assert drawn_image is detected_objects.plotted


def test_load_reports_failed_download_context(monkeypatch, tmp_path):
    predictor = object.__new__(MivoloPredictor)
    predictor.cache_path = str(tmp_path)
    monkeypatch.setattr(predictor_module, "simple_download", lambda **kwargs: False)
    monkeypatch.setattr(predictor_module, "public_oss_url", lambda path: f"https://models.example/{path}")

    with pytest.raises(RuntimeError) as exc_info:
        predictor.load()

    message = str(exc_info.value)
    assert "mivolo_imbd.pth.tar" in message
    assert str(tmp_path) in message
    assert "https://models.example/models/mivolo/mivolo_imbd.pth.tar" in message
