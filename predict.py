"""
download face_landmark and clone the stylegan-encoder repo first
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
git clone https://github.com/Puzer/stylegan-encoder
"""
import sys
sys.path.insert(0, "stylegan-encoder")
import tempfile
import dlib
from cog import BasePredictor, Path, Input

from ffhq_dataset.face_alignment import image_align
from ffhq_dataset.landmarks_detector import LandmarksDetector


PREDICTOR = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
LANDMARKS_DETECTOR = LandmarksDetector("shape_predictor_68_face_landmarks.dat")


class Predictor(BasePredictor):
    def setup(self):
        pass

    def predict(
        self,
        image: Path = Input(
            description="Input source image.",
        ),
    ) -> Path:
        out_path = Path(tempfile.mkdtemp()) / "output.png"
        align_image(str(image), str(out_path))
        return out_path


def align_image(raw_img_path, aligned_face_path):
    for i, face_landmarks in enumerate(
        LANDMARKS_DETECTOR.get_landmarks(raw_img_path), start=1
    ):
        image_align(raw_img_path, aligned_face_path, face_landmarks)
