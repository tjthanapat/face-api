from typing import Tuple, List
from dataclasses import dataclass

from deepface import DeepFace
import numpy as np


@dataclass
class FacialArea:
    x: int
    y: int
    w: int
    h: int


@dataclass
class FaceObj:
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
    confidence: float = 0.99,
) -> List[FaceObj]:
    """Detect and align faces in a given image.

    Parameters
    ----------
    img : array with shape (H, W, C)
    target_size : Tuple[int, int]
        Target size of cropped face region(s).
    confidence : float, optional
        Confidence used to filter detected faces, by default 0.99

    Returns
    -------
    face_objs : list of face objects
        Each face object contains face (cropped and aligned face image),
        facial_area (x, y, w, h) and confidence.
    """
    face_objs = DeepFace.extract_faces(
        img,
        target_size=target_size,
        detector_backend="mtcnn",
        enforce_detection=False,
    )
    sure_face_objs = [
        face_obj for face_obj in face_objs if face_obj["confidence"] >= confidence
    ]
    return sure_face_objs


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
    array
        Face feature vector
    """
    # Check if image's shape is the same as model's input shape.
    # if face_img.shape !== embedding_model.input_shape
    if normalize:
        face_img = _normalization(face_img)
    embedding = embedding_model.predict(np.expand_dims(face_img, axis=0))[0]
    return embedding


def _cosine(a: np.ndarray, b: np.ndarray):
    """Calculate cosine similarity between two given arrays with the same shape.

    Parameters
    ----------
    a : array
    b : array

    Returns
    -------
    float
        Cosine similarity
    """
    cosine_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return cosine_sim


def calculate_similarity(
    embedding_1: np.ndarray,
    embedding_2: np.ndarray,
) -> float:
    """Calculate cosine similarity between two given embeddings.

    Parameters
    ----------
    embedding_1 : array with shape (m,)
    embedding_2 : array with shape (m,)

    Returns
    -------
    float
        Cosine similarity
    """
    assert embedding_1.shape == embedding_2.shape, (
        "Two embeddings must have the same shape, "
        + f"{embedding_1.shape} and {embedding_2.shape}"
    )
    similarity = _cosine(embedding_1, embedding_2)
    return similarity
