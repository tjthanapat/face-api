import numpy as np
from PIL import Image
from io import BytesIO
from fastapi import UploadFile, HTTPException
import logging

from .custom_api_responses import API_STATUS_CODE


def _read_img_bytes(img_file) -> np.ndarray:
    img = np.array(
        Image.open(BytesIO(img_file)).convert("RGB"),
        dtype=np.uint8,
    )
    return img


async def read_img_file(img_file: UploadFile):
    if not img_file.filename.split(".")[-1].lower() in ("jpg", "jpeg", "png"):
        error_message = "Image file must be jpg or png format."
        logging.error(error_message)
        raise HTTPException(
            status_code=API_STATUS_CODE["NOT_SUPPORTED_IMAGE_FILE"],
            detail=error_message,
        )
    return _read_img_bytes(await img_file.read())


def normalize_img(img: np.ndarray) -> np.ndarray:
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
