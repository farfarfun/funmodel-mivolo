from funmodel.mivolo.predictor import MivoloPredictor
from funmodel.utils.image.transform import url_to_cvimg


def example_with_image():
    predictor = MivoloPredictor(draw=True, device="cpu", verbose=True)
    url = "https://nm-algo.linkheer.com/outfit/f2eff288546f8f331b754cbb5f90de1fe513edbc5a9da6dbfa01842a7b9dd623.jpg"
    result, out_im = predictor.predict(url_to_cvimg(url))
    print(result)


def example_with_capture():
    import cv2

    predictor = MivoloPredictor(draw=True, device="cpu", verbose=True)
    source = 0
    # source = 'http://30.204.102.21:4747/mjpegfeed'
    for result, frame, img in predictor.predict_capture(source=source):
        print(result)
        # cv2.imshow("capture", cv2.hconcat([frame, img]))
        cv2.imshow("capture", img)
        cv2.waitKey(1)


# example_with_image()
example_with_capture()
