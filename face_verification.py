from typing import Tuple, List
from dataclasses import dataclass

from deepface import DeepFace
import numpy as np

from utils import FacialArea


@dataclass
class DeepfaceFaceObj:
    face: np.ndarray
    facial_area: FacialArea
    confidence: float


# detection backends in DeepFace library
backends = [
    "opencv",
    "ssd",
    "dlib",
    "mtcnn",
    "retinaface",
    "mediapipe",
]


def detect(
    img: np.ndarray,
    target_size: Tuple[int, int],
) -> List[DeepfaceFaceObj]:
    """Detect and align faces in a given image.

    Parameters
    ----------
    img : array with shape (H, W, C)
    target_size : Tuple[int, int]
        Target size of cropped face region(s).

    Returns
    -------
    face_objs : list of face objects
        Each face object contains face (cropped and aligned face image),
        facial_area (x, y, w, h) and confidence.
    """
    try:
        face_objs = DeepFace.extract_faces(
            img,
            target_size=target_size,
            detector_backend="mtcnn",
        )
        face_objs_ = [
            dict(
                face=face_obj["face"],
                facial_area=face_obj["facial_area"],
                # DeepFace returns confidence in numpy.float64
                confidence=face_obj["confidence"].item(),
            )
            for face_obj in face_objs
        ]
        return face_objs_
    except ValueError:
        # DeepFace.extract_faces will raise exception when no face is detected.
        # logger.exception("message")
        return []


def _normalization(img: np.ndarray) -> np.ndarray:
    """Zero mean and unit variance normalizarion.

    Parameters
    ----------
    img : array

    Returns
    -------
    array
        Normalized image
    """
    mean, std = img.mean(), img.std()
    return (img - mean) / std


def embed(
    face_img: np.ndarray,
    embedding_model,
    normalize: bool = True,
) -> np.ndarray:
    """Embed face image, expecting single face.

    Parameters
    ----------
    face_img : array
    embedding_model : tensorflow keras model
    normalize : bool, optional
        If true, normalize image with zero mean and unit variance
        normalizarion, by default True

    Returns
    -------
    1D array
        Face feature vector
    """
    # Check if image's shape is the same as model's input shape.
    assert face_img.shape == embedding_model.input_shape[1:], (
        "Input image does not match model input shape. "
        + f"Image {face_img.shape} and model input shape {embedding_model.input_shape[1:]}"
    )
    face_img = face_img.astype(np.float32)  # ensure image's dtype
    if normalize:
        face_img = _normalization(face_img)
    embedding = embedding_model.predict(np.expand_dims(face_img, axis=0))[0]
    return embedding


def _cosine(a: np.ndarray, b: np.ndarray):
    """Calculate cosine similarity, range [-1, 1], between two given
    1D arrays sharing the same shape.

    Parameters
    ----------
    a : 1D array
    b : 1D array

    Returns
    -------
    float
        Cosine similarity
    """
    cosine_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    # Convert numpy float to native python float
    return cosine_sim.item()


def calculate_similarity(
    embedding_1: np.ndarray,
    embedding_2: np.ndarray,
) -> float:
    """Calculate cosine similarity, range [0, 1], between two given embeddings.

    Parameters
    ----------
    embedding_1 : 1D array with shape (m,)
    embedding_2 : 1D array with shape (m,)

    Returns
    -------
    float
        Cosine similarity
    """
    assert embedding_1.shape == embedding_2.shape, (
        "Two embeddings must have the same shape, "
        + f"{embedding_1.shape} and {embedding_2.shape}"
    )
    similarity = (_cosine(embedding_1, embedding_2) + 1) / 2
    return similarity
