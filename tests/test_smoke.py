# NOTE: This test is intentionally minimal.
#
# funmodel-mivolo ships as a `funmodel.mivolo` sub-package layered on top of
# the base `funmodel` distribution (both publish files under the shared
# `funmodel/` top-level namespace). Importing it pulls in a heavy dependency
# chain (torch, ultralytics, timm, fundrive, funget). No network/filesystem
# side effects happen at import time -- `MivoloPredictor.load()` only
# performs downloads when the class is instantiated, which this smoke test
# avoids.
import funmodel.mivolo
from funmodel.mivolo import MivoloPredictor


def test_import_funmodel_mivolo():
    assert funmodel.mivolo is not None


def test_import_mivolo_predictor_class():
    assert MivoloPredictor is not None
