import numpy as np
import pandas as pd
from typing import Union, List
from .custom_classes import RecognitionObj

def _calculate_similarities_with_db(
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


def recognize(
    embedding: np.ndarray,
    db_embeddings: pd.DataFrame,
    top_n: int = 5,
) -> List[RecognitionObj]:
    """Recognize face, comparing a given embedding with each embedding
    in db file. Embedding is vector with length m.

    Parameters
    ----------
    embedding : array with shape (m,)
    db_embeddings : dataframe with shape (n, 1 + m)
        Dataframe contains subject_id col and m embedding cols.
    top_n : int, optional
        Number of results ordered by confidence. 

    Returns
    -------
    list of result dicts
        Result dicts, each contains subject id and its confidence.
    """
    similarities = _calculate_similarities_with_db(embedding, db_embeddings.iloc[:, 1:])

    result = pd.concat([db_embeddings["subject_id"], similarities], axis=1)
    result.columns = ["subject_id", "confidence"]
    result.sort_values("confidence", ascending=False, inplace=True)
    result.reset_index(drop=True, inplace=True)

    result = result[:top_n]

    return result.to_dict("records")