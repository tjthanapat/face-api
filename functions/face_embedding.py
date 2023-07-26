import numpy as np
from .utilities import normalize_img

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
        face_img = normalize_img(face_img)
    embedding = embedding_model.predict(np.expand_dims(face_img, axis=0))[0]
    return embedding

