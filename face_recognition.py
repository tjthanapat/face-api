import numpy as np
import pandas as pd

import os
from pydantic import BaseModel
from typing import List, Union


class RecognitionResult(BaseModel):
    subject_id: str
    confidence: float


def add_embedding_into_db(
    db_embeddings_filepath: str,
    subject_id: str,
    embedding: np.ndarray,
):
    """Add embedding into db csv file. If give db file path does not exist,
    new file will be created.

    Parameters
    ----------
    db_embeddings_filepath : str
    subject_id : str
    embedding : array with shape (m,), same as other embedding in db
    """
    embedding = embedding.astype(float)  # ensure data type
    if not os.path.exists(db_embeddings_filepath):
        subject_arr = np.array([subject_id])
        db = pd.DataFrame(np.concatenate((subject_arr, embedding))[np.newaxis, ...])
        db.columns = ["subject_id"] + [i for i in range(len(embedding))]
        db.to_csv(db_embeddings_filepath, index=False)
    else:
        db = pd.read_csv(db_embeddings_filepath)
        assert (db.shape[1] - 1) == embedding.shape[0], (
            f"Embedding vector ({embedding.shape[0]},) does not share "
            + f"the same length with others in db ({db.shape[1] - 1},)."
        )
        subject_arr = np.array([subject_id])
        db.loc[len(db)] = np.concatenate((subject_arr, embedding)).ravel()
        db.to_csv(db_embeddings_filepath, index=False)


def calculate_similarities_with_db(
    embedding: np.ndarray,
    db_embeddings: Union[np.ndarray, pd.DataFrame],
) -> Union[np.ndarray, pd.Series]:
    """Calculate pairwise cosine similarities, ranging [0, 1],
    between an embedding with n db embeddings. Each embedding
    is vector with length m.

    Parameters
    ----------
    embedding : array with shape (m,)
    db_embeddings : array or dataframe with shape (n, m)

    Returns
    -------
    array or series with shape (n,)
        Cosine similarities
    """
    assert (
        embedding.shape == db_embeddings.shape[1:]
    ), "Embedding vectors must share the same length."

    # cosine_sim = A dot B / norm(A) * norm(B)
    dot_products = db_embeddings.dot(embedding)
    norm_embedding = np.linalg.norm(embedding)
    norm_db_embedding = np.linalg.norm(db_embeddings, axis=1)
    cosine_sim = dot_products / (norm_embedding * norm_db_embedding)

    # change cosine sim range from [-1,1] to [0,1]
    cosine_sim = (cosine_sim + 1) / 2

    return cosine_sim.astype(float)


def recognize_face(
    embedding: np.ndarray,
    db_embeddings_filepath: str,
    top_n: int = 5,
) -> List[RecognitionResult]:
    """Recognize face, comparing a given embedding with each embedding
    in db file. Embedding is vector with length m.

    Parameters
    ----------
    embedding : array with shape (m,)
    db_embeddings_filepath : str
        Path to db csv file.
    top_n : int, optional
        Number of results ordered by confidence. 

    Returns
    -------
    list of result dicts
        Result dicts, each contains subject id and its confidence.
    """
    db_embeddings = pd.read_csv(db_embeddings_filepath)
    similarities = calculate_similarities_with_db(embedding, db_embeddings.iloc[:, 1:])

    result = pd.concat([db_embeddings["subject_id"], similarities], axis=1)
    result.columns = ["subject_id", "confidence"]
    result.sort_values("confidence", ascending=False, inplace=True)
    result.reset_index(drop=True, inplace=True)

    result = result[:top_n]

    return result.to_dict("records")
